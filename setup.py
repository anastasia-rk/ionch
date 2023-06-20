# imports
import numpy as np
import scipy as sp
from scipy.interpolate import BSpline
from autograd import hessian, jacobian, grad
import matplotlib
from matplotlib import pyplot as plt
import time as tm
plt.rcParams['figure.figsize'] = (20,10)
plt.rcParams['figure.dpi'] = 400
plt.rcParams['axes.facecolor']='white'
plt.rcParams['savefig.facecolor']='white'
plt.style.use("ggplot")
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica"]})

## Definitions
def collocm(splinelist, tau):
    # collocation matrix for B-spline values (0-derivative)
    # inputs: splinelist - list of splines along one axis, tau - interval on which we wish to evaluate splines
    # outputs: collocation matrix
    mat = [[0] * len(tau) for _ in range(len(splinelist))]
    for i in range(len(splinelist)):
        mat[i][:] = splinelist[i](tau)
    return np.array(mat)


def ion_channel_model(t, x, theta):
    a, r = x[:2]
    *p, g = theta[:9]
    k1 = p[0] * np.exp(p[1] * V(t))
    k2 = p[2] * np.exp(-p[3] * V(t))
    k3 = p[4] * np.exp(p[5] * V(t))
    k4 = p[6] * np.exp(-p[7] * V(t))
    a_inf = k1 / (k1 + k2)
    tau_a = 1 / (k1 + k2)
    r_inf = k4 / (k3 + k4)
    tau_r = 1 / (k3 + k4)
    da = (a_inf - a) / tau_a
    dr = (r_inf - r) / tau_r
    return [da,dr]

def observation(t, x, theta):
    # I
    a, r = x[:2]
    *ps, g = theta[:9]
    return g * a * r * (V(t) - EK)

# get Voltage for time in ms
def V(t):
    return volts_intepolated(t / 1000)

# define the cost function for finding the smoothing coefficients
def cost_smoothing(spline_theta, theta, y, weight):
    coeffs_a, coeffs_r = np.split(spline_theta, 2)
    # collocation matrix is the fastest way to compute the whole curve at measurment points
    a = coeffs_a @ collocation
    r = coeffs_r @ collocation
    *ps, g = theta[:9]
    y_model = g * a * r * (volts_new - EK)
    # compute the data cost of the B-spline surface
    data_cost = np.transpose(y - y_model)@(y - y_model)
    ## model descrepancy
    tck_a = tuple([knots, coeffs_a, degree])
    tck_r = tuple([knots, coeffs_r, degree])
    dot_a = sp.interpolate.splev(times_q, tck_a, der=1)
    dot_r = sp.interpolate.splev(times_q, tck_r, der=1)
    fun_a = sp.interpolate.splev(times_q, tck_a, der=0)
    fun_r = sp.interpolate.splev(times_q, tck_r, der=0)
    # the derivatives must be put into an array
    dadr = []
    for it,t_q in enumerate(times_q):
        x = [fun_a[it],fun_r[it]]
        dadr.append(ion_channel_model(t_q, x, theta))
    rhs_theta = np.array(dadr)
    spline_deriv = np.transpose(np.array([dot_a, dot_r]))
    # compute the ode-fit cost of the B-spline surface
    ode_cost = np.trace(np.transpose(spline_deriv - rhs_theta)@(spline_deriv - rhs_theta))
    # return the regularised cost
    return data_cost + weight * ode_cost

def cost_simple(spline_theta):
    coeffs_a, coeffs_r = np.split(spline_theta, 2)
    # collocation matrix is the fastest way to compute the whole curve at measurment points
    a = coeffs_a @ collocation
    r = coeffs_r @ collocation
    *ps, g = p_true[:9]
    y_model = g * a * r * (volts_new - EK)
    # compute the data cost of the B-spline surface
    data_cost = np.transpose(current - y_model)@(current - y_model)
    ## model descrepancy
    tck_a = tuple([knots, coeffs_a, degree])
    tck_r = tuple([knots, coeffs_r, degree])
    dot_a = sp.interpolate.splev(times_q, tck_a, der=1)
    dot_r = sp.interpolate.splev(times_q, tck_r, der=1)
    fun_a = sp.interpolate.splev(times_q, tck_a, der=0)
    fun_r = sp.interpolate.splev(times_q, tck_r, der=0)
    # the derivatives must be put into an array
    dadr = []
    for it,t_q in enumerate(times_q):
        x = [fun_a[it],fun_r[it]]
        dadr.append(ion_channel_model(t_q, x, p_true))
    rhs_theta = np.array(dadr)
    spline_deriv = np.transpose(np.array([dot_a, dot_r]))
    # compute the ode-fit cost of the B-spline surface
    ode_cost = np.trace(np.transpose(spline_deriv - rhs_theta)@(spline_deriv - rhs_theta))
    # return the regularised cost
    return data_cost + lambd * ode_cost

def jac_of_cost(spline_theta):
    input_v = spline_theta
    jcb = jacobian(cost_simple, input=input_v)
    return jcb

def func(x):
   return x**3 + 4*x -10

def func_jac(x):
    return(jacobian(func,x))

# main
if __name__ == '__main__':
    #  load the voltage data:
    volts = np.genfromtxt("./protocol-staircaseramp.csv", skip_header=1, dtype=float, delimiter=',')
    #  check when the voltage jumps
    # read the times and valued of voltage clamp
    volt_times, volts = np.genfromtxt("./protocol-staircaseramp.csv", skip_header=1, dtype=float, delimiter=',').T
    switchpoints = np.abs(np.diff(volts, n=2)) > 1e-6
    # interpolate with smaller time step (milliseconds)
    volts_intepolated = sp.interpolate.interp1d(volt_times, volts, kind='previous')


    # tlim = [0, int(volt_times[-1]*1000)]
    tlim = [0,5000]
    times = np.linspace(*tlim, tlim[-1])
    # get time points to compute the fit to ODE cost
    times_q = times[[(it % 5 == 0) for it in range(len(times))]]
    volts_new = V(times)
    d2v_dt2 = np.diff(volts_new, n=2)
    switchpoints_new = np.abs(d2v_dt2) > 1e-6
    ## uncomment this to see the second derivative of the voltage clamp
    # fig, ax = plt.subplots()
    # ax.plot(times[2:], d2v_dt2, 'b', label='Voltage clamp')
    # ax.legend(fontsize=14)
    # ax.set_xlabel('times, s')
    # ax.set_ylabel('2nd derivative')
    # plt.savefig('Figures/2nd_derivative.png')

    # get times of jumps and a B-spline knot sequence
    nPoints_closest = 4 # the number of points from each jump where knots are placed at every timestep
    nPoints_around_jump = 52
    nPoints_between_jumps = 3
    step_between_knots = 13 # this is the index step between knots around the jump
    # get the times of all jumps
    jump_times = [tlim[0]] + [i for i, x in enumerate(switchpoints_new) if x] + [tlim[-1]]  # get indeces of all the switchpoints, add t0 and tend
    distance_lists = [[abs(num - index) for num in range(len(times))] for index in
                      jump_times]  # compute distance between eact time and time of jump
    min_distances = [min(tpl) for tpl in
                     zip(*distance_lists)]  # get minimal distance to the nearest jump for each timepoint
    knots_jump = [(x <= nPoints_closest) or ((nPoints_closest <= x <= nPoints_around_jump) and (x % step_between_knots == 0)) for x in
                  min_distances]  # create a knot sequence that has higher density of knots around each jump
    # some ad hoc solutions there - take every other point near the jump
    # to this we then need to add additional coarse grid of knots between two jumps:
    knot_times = [i for i, x in enumerate(knots_jump) if x] # convert to numeric array again
    knots_all = knot_times
    for iKnot, timeKnot in enumerate(knot_times[:-1]):
        if knot_times[iKnot+1] - timeKnot > step_between_knots:
            # create 6 evenly spaced points and drop start and end - those are already in the grid
            knots_between_jumps = np.rint(np.linspace(timeKnot, knot_times[iKnot+1],num=nPoints_between_jumps+2)[1:-1]).astype(int)
            # add indeces to the list
            knots_all = knots_all + list(knots_between_jumps)
    knots_all.sort() # sort list in ascending order - this is done inplace!
    # knots_boolean = [True if ele in set(knots_all) else False for ele in np.arange(len(times))] # get boolean index for all times
    knots_boolean = np.isin(np.arange(len(times)),knots_all) # make a boolean mask for the time points


    # fig, axes = plt.subplots(2,1)
    # axes[0].plot(volt_times, volts, 'b', label='Voltage clamp')
    # axes[0].plot(volt_times[2:][switchpoints], volts[2:][switchpoints], 'r.',label='Swithchpoints')
    # axes[0].legend(fontsize=14)
    # axes[0].set_xlabel('times, s')
    # axes[0].set_ylabel('voltage, mV')
    # axes[0].set_xlim([volt_times[0], volt_times[-1]])
    # axes[1].plot(times, volts_new, 'b')
    # axes[1].plot(times[2:][switchpoints_new], volts_new[2:][switchpoints_new], 'r.')
    # axes[1].set_xlabel('times, ms')
    # axes[1].set_ylabel('voltage, mV')
    # axes[1].set_xlim(tlim)
    # plt.tight_layout()
    # plt.savefig('Figures/voltage_interpolated.png')

    # parameter values for the model
    EK = -80
    p_true = [2.26e-4, 0.0699, 3.45e-5, 0.05462, 0.0873, 8.91e-3, 5.15e-3, 0.03158, 0.1524]
    # initialise and solve ODE
    x0 = [0, 1]
    solution = sp.integrate.solve_ivp(ion_channel_model, tlim, x0, args=[p_true], dense_output=True)
    x_ar = solution.sol(times)
    current = observation(times, x_ar, p_true)

    # # plot three states and the output
    # fig, axes = plt.subplots(2, 2)
    # axes[0, 0].plot(times, x_ar[0,:], 'b')
    # axes[0, 0].plot(times[2:][switchpoints_new], x_ar[0,2:][switchpoints_new], 'r.')
    # axes[0, 0].set_xlabel('times, ms')
    # axes[0, 0].set_ylabel('a gating variable')
    # axes[0, 0].set_xlim(tlim)
    # axes[0, 1].plot(times, x_ar[1,:], 'b')
    # axes[0, 1].plot(times[2:][switchpoints_new], x_ar[1,2:][switchpoints_new], 'r.')
    # axes[0, 1].set_xlabel('times, ms')
    # axes[0, 1].set_ylabel('r gating variable')
    # axes[0, 1].set_xlim(tlim)
    # axes[1, 0].plot(times, volts_new, 'b')
    # axes[1, 0].plot(times[2:][switchpoints_new], volts_new[2:][switchpoints_new], 'r.')
    # axes[1, 0].set_xlabel('times, ms')
    # axes[1, 0].set_ylabel('voltage, mV')
    # axes[1, 0].set_xlim(tlim)
    # axes[1, 1].plot(times, current, 'b')
    # axes[1, 1].plot(times[2:][switchpoints_new], current[2:][switchpoints_new], 'r.')
    # axes[1, 1].set_xlabel('times, ms')
    # axes[1, 1].set_ylabel('Current, A')
    # axes[1, 1].set_xlim(tlim)
    # plt.tight_layout()
    # plt.savefig('Figures/model_states_output.png')

    # make a list of states
    stateNames = ['a','r','V, mV']
    states = [x_ar[0], x_ar[1], volts_new]
    # times = times[2:]
    nStates = len(states)
    fig, axes = plt.subplots(nStates,1, sharex=True)
    for iState in range(nStates):
        state = states[iState]
        tck = sp.interpolate.splrep(times[knots_boolean], state[knots_boolean], s=0, k=3)
        y_fit = sp.interpolate.BSpline(*tck)(times)
        ax = axes.flatten()[iState]
        ax.plot(times, state, '--k',alpha=0.7,label='true')
        ax.plot(times[knots_boolean], state[knots_boolean],'.b',label='knots')
        ax.plot(times,y_fit,label='fit')
        ax.set_ylabel(stateNames[iState])
        # ax.set_xlim(3500, 5000)
    axes.flatten()[iState].legend(fontsize=14, loc='best') # only put legend into last axes
    axes.flatten()[iState].set_xlabel('time, ms')
    plt.tight_layout()
    figName = 'Figures/b_splines_native_irreg_grid_' + str(nPoints_around_jump) + '_near_jump_' + str(nPoints_between_jumps) +'_between.png'
    plt.savefig(figName)

    # build the collocation matrix using the defined knot structure
    degree = 3
    fig, ax = plt.subplots()
    outer = [knots_all[0],knots_all[0], knots_all[0], knots_all[-1], knots_all[-1], knots_all[-1]]
    outer_y = []
    knots = np.insert(outer, 3, knots_all)  # create knots for which we want to sbuild splines
    coeffs = np.zeros(len(knots) - degree - 1)  # number of splines will depend on the knot order
    spl_ones = BSpline(knots, np.ones_like(coeffs), degree)
    tau = np.arange(knots[0], knots[-1]+1)
    splinest = [None] * len(coeffs)  # the grid of indtividual splines is required to generate a collocation matrix
    # fig, ax = plt.subplots()
    for i in range(len(coeffs)):
        tau_current = np.arange(knots[i], knots[i+4])
        coeffs[i] = 1
        splinest[i] = BSpline(knots, coeffs.copy(), degree, extrapolate=False)  # create a spline that only has one non-zero coeff
        # ax.plot(tau_current, splinest[i](tau_current), lw=0.5, alpha=0.7)
        coeffs[i] = 0
    collocation = collocm(splinest, tau) #create a collocation matrix for that interval
    ####################################################################################################################
    # uncomment this to plot the grid of splines with coeff 1 each
    # ax.plot(np.ones_like(coeffs) @ collocation, '--r', lw=0.5, alpha=0.7, label='B-spline curve')
    # # draw lines indicating the jumps
    # for _, jump in enumerate(jump_times):
    #     ax.axvline(x=jump, ls='--', color='k', linewidth=0.5, alpha=0.7,)
    # ax.grid(True)
    # ax.set_ylabel('B-spline grid for all time points')
    # ax.set_xlabel('times, ms')
    # ax.set_xlim(tlim)
    # # ax.set_xlim(0,2500)
    # ax.legend(fontsize=14,loc='upper right')
    # plt.tight_layout()
    # plt.savefig('Figures/Bspline_grid.png')

    # fit the B-splines coeff using direct LS
    # fig, axes = plt.subplots(2,1,sharex=True)
    all_coeffs = []
    for iState in range(nStates-1):
        ax = axes.flatten()[iState]
        state = states[iState]
        coeffs_ls = np.dot((np.dot(np.linalg.inv(np.dot(collocation, collocation.T)), collocation)), state)
        all_coeffs.append(coeffs_ls)
        coeffs = np.zeros_like(coeffs_ls)
    ####################################################################################################################
    #     # uncomment this to test formation of the splines and their derivatives
    #     tck_a = tuple([knots, coeffs_ls, degree])
    #     dot_a = sp.interpolate.splev(times, tck_a, der=1)
    #     fun_a = sp.interpolate.splev(times, tck_a, der=0)
    #     ax.plot(times, state, 'k', lw=1, alpha=0.7, label='true')
    #     ax.plot(times, dot_a,'--b',lw=1,label='Derivative')
    #     ax.plot(times, fun_a,'--r',lw=1,label='Curve')
    #     ax.set_ylabel(stateNames[iState])
    #     ax.set_xlim(tlim)
    # ax.legend(fontsize=14, loc='best')  # only put legend into last axes
    # ax.set_xlabel('time, ms')
    # figName = 'Figures/check_derivatives.png'
    # plt.savefig(figName)

    #  check the derivatives
    coeffs_a, coeffs_r = all_coeffs
    tck_a = tuple([knots, coeffs_a, degree])
    tck_r = tuple([knots, coeffs_r, degree])
    dot_a = sp.interpolate.splev(times, tck_a, der=1)
    dot_r = sp.interpolate.splev(times, tck_r, der=1)
    fun_a = sp.interpolate.splev(times, tck_a, der=0)
    fun_r = sp.interpolate.splev(times, tck_r, der=0)
    dadr = []
    for it, t_q in enumerate(times):
        x = [fun_a[it], fun_r[it]]
        dadr.append(ion_channel_model(t_q, x, p_true))
    rhs_theta = np.array(dadr)
    fig, axes = plt.subplots(2, 2, sharex=True)
    y_labels = ['a','$\dot{a}$','r','$\dot{r}$']
    axes[0, 0].plot(x_ar[0], '-k', label='true')
    axes[0, 0].plot(fun_a, '--r', label='B-splines')
    axes[1, 0].plot(x_ar[1], '-k', label='true')
    axes[1, 0].plot(fun_r, '--r', label='B-splines')
    axes[0, 1].plot(rhs_theta[:,0], '-k', label='RHS')
    axes[0, 1].plot(dot_a, '--r', label='B-spline derivative')
    axes[1, 1].plot(rhs_theta[:,1], '-k', label='RHS')
    axes[1, 1].plot(dot_r, '--r', label='B-spline derivative')
    for iAx, ax in enumerate(axes.flatten()):
        ax.legend(fontsize=14, loc='upper right')
        ax.set_ylabel(y_labels[iAx])
    plt.tight_layout()
    plt.savefig('Figures/LS_b_splin_grid.png')
    ####################################################################################################################
    ## Uncomment this to plot the B-spline functions and approximations
    #     for i in range(len(coeffs)):
    #         tau_current = np.arange(knots[i], knots[i + 4])
    #         coeffs[i] = coeffs_ls[i]
    #         splinest[i] = BSpline(knots, coeffs, degree,
    #                               extrapolate=False)  # create a spline that only has one non-zero coeff
    #         ax.plot(tau_current, splinest[i](tau_current), lw=0.5, alpha=0.7)
    #         coeffs[i] = 0
    #     ax.plot(times, state, '-k', lw=0.5, alpha=0.7, label='true')
    #     ax.plot(coeffs_ls @ collocation, '--r', lw=1, alpha=0.7, label='B-spline curve')
    #     ax.set_ylabel(stateNames[iState])
    #     # ax.set_xlim(tlim)
    #     ax.set_xlim(0,2500)
    # ax.legend(fontsize=14, loc='best')  # only put legend into last axes
    # ax.set_xlabel('time, ms')
    # figName = 'Figures/example_bspl_fit_cropped.png'
    # plt.savefig(figName)
    ##
    ## optimisation accroding to Ramsey&Hooker cost: standard python minimizer with Broyden-Fletcher-Goldfarb-Shanno gradient descent
    lambd = 100
    coeffs_init = np.ones(len(coeffs)*2,)
    tic = tm.time()
    res = sp.optimize.minimize(cost_smoothing, coeffs_init, args=(p_true, current, lambd), method='BFGS',options={'disp': True})
    ## try with autograd
    # jacob = jac_of_cost(np.hstack(all_coeffs))
    # res = sp.optimize.minimize(cost_simple, coeffs_init, method='BFGS',options={'disp': True})
    toc = tm.time()
    print('Time elapsed = ' + str(tic-toc) + ' s')
    ####################################################################################################################
    ## from here onwards the initial idea of pieacewise support between jumps
    # jump_times = [0, 228, 699]
    # degree = 3
    # nKnots = 4 # to get 10 splines for each interval
    # coeffs_all = []
    # collocations = []
    # fig, ax = plt.subplots()
    # for iJump in range(2):
    #     outer = [jump_times[iJump] - 30,jump_times[iJump] - 20, jump_times[iJump] - 10, jump_times[iJump+1] + 10, jump_times[iJump+1] + 20, jump_times[iJump+1] + 30]
    #     knots = np.insert(outer, 3, np.linspace(jump_times[iJump], jump_times[iJump+1],
    #                                             nKnots))  # create knots for which we want to sbuild splines
    #     coeffs = np.zeros(len(knots) - degree - 1)  # number of splines will depend on the knot order
    #     spl_ones = BSpline(knots, np.ones_like(coeffs), degree)
    #     tau = np.arange(jump_times[iJump], jump_times[iJump+1])
    #     splinest = [None] * len(coeffs)  # the grid of indtividual splines is required to generate a collocation matrix
    #     for i in range(len(coeffs)):
    #         coeffs[i] = iJump+1
    #         splinest[i] = BSpline(knots, coeffs.copy(), degree, extrapolate=False)  # create a spline that only has one non-zero coeff
    #         ax.plot(tau, splinest[i](tau), lw=0.5, alpha=0.7)
    #         coeffs[i] = 0
    #     collm_t = collocm(splinest, tau) #create a collocation matrix for that interval
    #     #     store stuff
    #     coeffs_all.append(np.ones_like(coeffs)*(iJump+1))
    #     collocations.append(collm_t)
    # # try entire matrix
    # all_coeffs = np.concatenate(coeffs_all, axis=0)
    # all_matrices = sp.linalg.block_diag(*collocations)
    # ax.plot(all_coeffs @ all_matrices, '--r', lw=3, label='B-spline curve')
    # ax.axvline(x=jump_times[0], color='b')
    # ax.axvline(x=jump_times[1], color='b')
    # ax.axvline(x=jump_times[2], color='b')
    # ax.grid(True)
    # ax.set_ylabel('B-spline grid between two points')
    # ax.set_xlabel('times, ms')
    # ax.legend(fontsize=14,loc='upper right')
    # plt.tight_layout()
    # plt.savefig('Figures/Bspline_grid_w_jump.png')
    #
    #
    # # test a B-spline fit to flat surface
    # degree = 3
    # # knots = 100*np.array([-10.5, -7, -3.5, 0, 3.5, 7, 10.5, 14, 17.5])
    # knots = 100 * np.array([-15, -10, -5, 0, .5, 5, 10, 15, 20])
    # xx = np.linspace(0, 500, 100)
    # coeffs = np.zeros(len(knots)-degree-1)
    # spl_ones = BSpline(knots, np.ones_like(coeffs), degree)
    # splinest = [None] * len(coeffs)
    # fig, ax = plt.subplots()
    # for i in range(len(coeffs)):
    #     coeffs[i] = 1
    #     splinest[i] = BSpline(knots, coeffs.copy(), degree, extrapolate=False)
    #     ax.plot(xx, splinest[i](xx), lw=2, alpha=0.7, label=str(i))
    #     coeffs[i] = 0
    # ax.plot(xx, spl_ones(xx), 'k', lw=3, label='Curve')
    # ax.grid(True)
    # ax.set_ylabel('B-spline grid')
    # ax.set_xlabel('Time, h')
    # ax.legend(fontsize=14)
    # # ax.set_xlim([0, 800])
    # plt.tight_layout()
    # plt.show()
    # # plt.savefig('Figures/Bspline_grid_time.png')