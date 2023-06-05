# imports
import numpy as np
import scipy as sp
from scipy.interpolate import BSpline
import matplotlib
from matplotlib import pyplot as plt
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


def ion_channel_model(t, x, p):
    a, r = x[:2]
    *ps, g = p[:9]
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

def observation(t, x, p):
    # I
    a, r = x[:2]
    *ps, g = p[:9]
    return g * a * r * (V(t) - EK)

# get Voltage for time in ms
def V(t):
    return volts_intepolated(t / 1000)

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


    tlim = [0, int(volt_times[-1]*1000)]
    times = np.linspace(*tlim, tlim[-1])
    volts_new = V(times)
    d2v_dt2 = np.diff(volts_new, n=2)
    switchpoints_new = np.abs(d2v_dt2) > 1e-6
    fig, ax = plt.subplots()
    ax.plot(volt_times, volts, 'b', label='Voltage clamp')
    ax.plot(volt_times[2:][switchpoints_new], volts[2:][switchpoints_new], 'r.', label='Swithchpoints')
    ax.legend(fontsize=14)
    ax.set_xlabel('times, s')
    ax.set_ylabel('2nd derivative')
    plt.savefig('Figures/2nd_derivative.png')

    fig, axes = plt.subplots(2,1)
    axes[0].plot(volt_times, volts, 'b', label='Voltage clamp')
    axes[0].plot(volt_times[2:][switchpoints], volts[2:][switchpoints], 'r.',label='Swithchpoints')
    axes[0].legend(fontsize=14)
    axes[0].set_xlabel('times, s')
    axes[0].set_ylabel('voltage, mV')
    axes[0].set_xlim([volt_times[0], volt_times[-1]])
    axes[1].plot(times, volts_new, 'b')
    axes[1].plot(times[2:][switchpoints_new], volts_new[2:][switchpoints_new], 'r.')
    axes[1].set_xlabel('times, ms')
    axes[1].set_ylabel('voltage, mV')
    axes[1].set_xlim(tlim)
    plt.tight_layout()
    plt.savefig('Figures/voltage_interpolated.png')

    # parameter values for the model
    EK = -80
    p_true = [2.26e-4, 0.0699, 3.45e-5, 0.05462, 0.0873, 8.91e-3, 5.15e-3, 0.03158, 0.1524]
    # initialise and solve ODE
    y0 = [0, 1]
    solution = sp.integrate.solve_ivp(ion_channel_model, tlim, y0, args=[p_true], dense_output=True)
    x_ar = solution.sol(times)
    current = observation(times, x_ar, p_true)

    # plot three states and the output
    fig, axes = plt.subplots(2, 2)
    axes[0, 0].plot(times, x_ar[0,:], 'b')
    axes[0, 0].plot(times[2:][switchpoints_new], x_ar[0,2:][switchpoints_new], 'r.')
    axes[0, 0].set_xlabel('times, ms')
    axes[0, 0].set_ylabel('a gating variable')
    axes[0, 0].set_xlim(tlim)
    axes[0, 1].plot(times, x_ar[1,:], 'b')
    axes[0, 1].plot(times[2:][switchpoints_new], x_ar[1,2:][switchpoints_new], 'r.')
    axes[0, 1].set_xlabel('times, ms')
    axes[0, 1].set_ylabel('r gating variable')
    axes[0, 1].set_xlim(tlim)
    axes[1, 0].plot(times, volts_new, 'b')
    axes[1, 0].plot(times[2:][switchpoints_new], volts_new[2:][switchpoints_new], 'r.')
    axes[1, 0].set_xlabel('times, ms')
    axes[1, 0].set_ylabel('voltage, mV')
    axes[1, 0].set_xlim(tlim)
    axes[1, 1].plot(times, current, 'b')
    axes[1, 1].plot(times[2:][switchpoints_new], current[2:][switchpoints_new], 'r.')
    axes[1, 1].set_xlabel('times, ms')
    axes[1, 1].set_ylabel('Current, A')
    axes[1, 1].set_xlim(tlim)
    plt.tight_layout()
    plt.savefig('Figures/model_states_output.png')

    # make a list of states
    states = [x_ar[0,2:], x_ar[1,2:], volts_new[2:]]
    times = times[2:]
    nStates = len(states)
    fig, axes = plt.subplots(nStates,1)
    for iState in range(nStates):
        state = states[iState]
        tck = sp.interpolate.splrep(times[switchpoints_new], state[switchpoints_new], s=0, k=3)
        y_fit = sp.interpolate.BSpline(*tck)(times)
        ax = axes.flatten()[iState]
        ax.plot(times,state,'.b')
        ax.plot(times,y_fit)
    plt.tight_layout()
    plt.savefig('Figures/b_splines_native.png')

    # # use python native b-splines
    # get the times of all jump
    jump_times = [tlim[0]] + [i for i, x in enumerate(switchpoints_new) if x] + [tlim[-1]]  # get indeces of all the switchpoints, add t0 and tend
    # degree = 3
    # nKnots = 10 # to get 10 splines for each interval
    # outer = [jump_times[0] - 20, jump_times[0] - 10, jump_times[1] + 10, jump_times[1] + 20]
    # knots = np.insert(outer, 2, np.linspace(jump_times[0],jump_times[1],nKnots)) # create knots for which we want to sbuild splines
    # coeffs = np.zeros(len(knots) - degree - 1) # number of splines will depend on the knot order
    # spl_ones = BSpline(knots, np.ones_like(coeffs), degree)
    # tau = np.arange(jump_times[0],jump_times[1])
    # # plot all spines in between the first two switchpoints
    # fig, ax = plt.subplots()
    # splinest = [None] * len(coeffs) # the grid of indtividual splines is required to generate a collocation matrix
    # for i in range(len(coeffs)):
    #     coeffs[i] = 1
    #     splinest[i] = BSpline(knots, coeffs.copy(), degree, extrapolate=False) # create a spline that only has one non-zero coeff
    #     ax.plot(tau, splinest[i](tau), lw=2, alpha=0.7, label=str(i))
    #     coeffs[i] = 0
    # ax.plot(tau, spl_ones(tau), 'k', lw=3, label='Curve')
    # collm_t = collocm(splinest, tau)
    # ax.plot(tau, np.ones_like(coeffs) @ collm_t, '--r', lw=3, label='Collocation method')
    # ax.axvline(x=jump_times[0], color='b', label='t =' + str(jump_times[0]) + 'ms')
    # ax.axvline(x=jump_times[1], color='b', label='t =' + str(jump_times[1]) + 'ms')
    # ax.grid(True)
    # ax.set_ylabel('B-spline grid between two points')
    # ax.set_xlabel('times, ms')
    # ax.legend(fontsize=14)
    # plt.tight_layout()
    # plt.savefig('Figures/Bspline_grid_between_jumps.png')

    # try with a switch
    jump_times = [0, 228, 699]
    degree = 3
    nKnots = 4 # to get 10 splines for each interval
    coeffs_all = []
    collocations = []
    fig, ax = plt.subplots()
    for iJump in range(2):
        outer = [jump_times[iJump] - 30,jump_times[iJump] - 20, jump_times[iJump] - 10, jump_times[iJump+1] + 10, jump_times[iJump+1] + 20, jump_times[iJump+1] + 30]
        knots = np.insert(outer, 3, np.linspace(jump_times[iJump], jump_times[iJump+1],
                                                nKnots))  # create knots for which we want to sbuild splines
        coeffs = np.zeros(len(knots) - degree - 1)  # number of splines will depend on the knot order
        spl_ones = BSpline(knots, np.ones_like(coeffs), degree)
        tau = np.arange(jump_times[iJump], jump_times[iJump+1])
        splinest = [None] * len(coeffs)  # the grid of indtividual splines is required to generate a collocation matrix
        for i in range(len(coeffs)):
            coeffs[i] = iJump+1
            splinest[i] = BSpline(knots, coeffs.copy(), degree, extrapolate=False)  # create a spline that only has one non-zero coeff
            ax.plot(tau, splinest[i](tau), lw=2, alpha=0.7)
            coeffs[i] = 0
        collm_t = collocm(splinest, tau) #create a collocation matrix for that interval
        #     store stuff
        coeffs_all.append(np.ones_like(coeffs)*(iJump+1))
        collocations.append(collm_t)
    # try entire matrix
    all_coeffs = np.concatenate(coeffs_all, axis=0)
    all_matrices = sp.linalg.block_diag(*collocations)
    ax.plot(all_coeffs @ all_matrices, '--r', lw=3, label='Collocation method')
    ax.axvline(x=jump_times[0], color='b')
    ax.axvline(x=jump_times[1], color='b')
    ax.axvline(x=jump_times[2], color='b')
    ax.grid(True)
    ax.set_ylabel('B-spline grid between two points')
    ax.set_xlabel('times, ms')
    ax.legend(fontsize=14)
    plt.tight_layout()
    plt.savefig('Figures/Bspline_grid_w_jump.png')






    # test a B-spline fit to flat surface
    degree = 3
    # knots = 100*np.array([-10.5, -7, -3.5, 0, 3.5, 7, 10.5, 14, 17.5])
    knots = 100 * np.array([-15, -10, -5, 0, .5, 5, 10, 15, 20])
    xx = np.linspace(0, 500, 100)
    coeffs = np.zeros(len(knots)-degree-1)
    spl_ones = BSpline(knots, np.ones_like(coeffs), degree)
    splinest = [None] * len(coeffs)
    fig, ax = plt.subplots()
    for i in range(len(coeffs)):
        coeffs[i] = 1
        splinest[i] = BSpline(knots, coeffs.copy(), degree, extrapolate=False)
        ax.plot(xx, splinest[i](xx), lw=2, alpha=0.7, label=str(i))
        coeffs[i] = 0
    ax.plot(xx, spl_ones(xx), 'k', lw=3, label='Curve')
    ax.grid(True)
    ax.set_ylabel('B-spline grid')
    ax.set_xlabel('Time, h')
    ax.legend(fontsize=14)
    # ax.set_xlim([0, 800])
    plt.tight_layout()
    plt.show()
    # plt.savefig('Figures/Bspline_grid_time.png')