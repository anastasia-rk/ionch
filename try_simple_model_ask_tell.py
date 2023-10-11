# imports
import matplotlib.pyplot as plt
import numpy as np

from setup import *
import pints
import sys
import pickle as pkl
plt.ioff()
matplotlib.use('AGG')

# definitions
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

# # Only consider A all params in log scale
# def ion_channel_model_one_state(t, x, theta):
#     # call the model with a smaller number of unknown parameters and one state known
#     a = x
#     v = V(t)
#     k1 =  np.exp(theta[0] + np.exp(theta[1]) * v)
#     k2 =  np.exp(theta[2] -np.exp(theta[3]) * v)
#     a_inf = k1 / (k1 + k2)
#     tau_a = 1 / (k1 + k2)
#     da = (a_inf - a) / tau_a
#     return da

# # only consider R (IN DECIMAL SCALE)
# def ion_channel_model_one_state(t, x, theta):
#     # call the model with a smaller number of unknown parameters and one state known
#     r = x
#     v = V(t)
#     k3 = theta[0] * np.exp(8.91e-3 * v)
#     k4 = theta[1] * np.exp(-0.03158 * v)
#     r_inf = k4 / (k3 + k4)
#     tau_r = 1 / (k3 + k4)
#     dr = (r_inf - r) / tau_r
#     return dr

# try log space on a parameters
def ion_channel_model_one_state(t, x, theta):
    # call the model with a smaller number of unknown parameters and one state known
    r = x
    v = V(t)
    k3 =  np.exp(theta[0] + 8.91e-3 * v)
    k4 =  np.exp(theta[1] - 0.03158 * v)
    r_inf = k4 / (k3 + k4)
    tau_r = 1 / (k3 + k4)
    dr = (r_inf - r) / tau_r
    return dr

def observation(t, x, theta):
    # I
    a, r = x[:2]
    *ps, g = theta[:9]
    return g * a * r * (V(t) - EK)
# get Voltage for time in ms
def V(t):
    return volts_intepolated((t)/ 1000)

# main
if __name__ == '__main__':
    #  load the voltage data:
    volts = np.genfromtxt("./protocol-staircaseramp.csv", skip_header=1, dtype=float, delimiter=',')
    #  check when the voltage jumps
    # read the times and valued of voltage clamp
    volt_times, volts = np.genfromtxt("./protocol-staircaseramp.csv", skip_header=1, dtype=float, delimiter=',').T
    # interpolate with smaller time step (milliseconds)
    volts_intepolated = sp.interpolate.interp1d(volt_times, volts, kind='previous')

    # tlim = [0, int(volt_times[-1]*1000)]
    tlim = [0, 6100]
    times = np.linspace(*tlim, tlim[-1])
    # define a region of interest - we will need this to preserve the
    # trajectories of states given the full clamp and initial position, while
    ROI_start = 3500
    ROI_end = tlim[-1]
    ROI = range(ROI_start,ROI_end)
    # get time points to compute the fit to ODE cost
    times_roi = times[ROI_start:ROI_end]
    times_quad = np.linspace(times_roi[0], times_roi[-1],num=2*len(ROI)) # set up time nodes for quadrature integration
    volts_new = V(times)
    d2v_dt2 = np.diff(volts_new, n=2)
    dv_dt = np.diff(volts_new)
    der1_nonzero = np.abs(dv_dt) > 1e-6
    der2_nonzero = np.abs(d2v_dt2) > 1e-6
    switchpoints = [a and b for a, b in zip(der1_nonzero, der2_nonzero)]
    # ignore everything outside of the region of iterest
    switchpoints_roi = switchpoints[ROI_start:ROI_end]

    ## Generate the synthetic data
    # parameter values for the model
    EK = -80
    p_true = [2.26e-4, 0.0699, 3.45e-5, 0.05462, 0.0873, 8.91e-3, 5.15e-3, 0.03158, 0.1524]
    *p, g = p_true
    # initialise and solve ODE
    x0 = [0, 1]
    # solve initial value problem
    solution = sp.integrate.solve_ivp(ion_channel_model, tlim, x0, args=[p_true], dense_output=True,method='LSODA',rtol=1e-8,atol=1e-8)
    x_ar = solution.sol(times_roi)
    current_true = observation(times_roi, x_ar, p_true)

    # # use a as unknown state
    # theta_true = [np.log(2.26e-4), np.log(0.0699), np.log(3.45e-5), np.log(0.05462)]
    # inLogScale = True
    # param_names = ['p_1','p_2','p_3','p_4']
    # a0 = [0]
    # solution_a = sp.integrate.solve_ivp(ion_channel_model_one_state, tlim, a0, args=[theta_true], dense_output=True, method='LSODA',
    #                                   rtol=1e-8, atol=1e-8)
    # state_hidden_true = solution_a.sol(times_roi)
    # state_known = x_ar[1,:] # assume that we know r
    # state_name = 'a'

    # use r as unknown state
    # theta_true = [0.0873, 5.15e-3]
    # inLogScale = False
    theta_true = [np.log(0.0873), np.log(5.15e-3)]
    inLogScale = True
    param_names = ['p_5','p_7']
    r0 = [1]
    solution_r = sp.integrate.solve_ivp(ion_channel_model_one_state, tlim, r0, args=[theta_true], dense_output=True,
                                        method='LSODA',
                                        rtol=1e-8, atol=1e-10)
    state_hidden_true = solution_r.sol(times_roi)
    state_known = x_ar[0, :]  # assume that we know r
    state_name = 'r'

    # # direct all output to file
    # stdoutOrigin = sys.stdout
    # sys.stdout = open("log_"+state_name+".txt", "w")
    ####################################################################################################################
    ## B-spline representation setup
    # set times of jumps and a B-spline knot sequence
    nPoints_closest = 24  # the number of points from each jump where knots are placed at the finest grid
    nPoints_between_closest = 12  # step between knots at the finest grid
    nPoints_around_jump = 48  # the time period from jump on which we place medium grid
    step_between_knots = 48  # this is the step between knots around the jump in the medium grid
    nPoints_between_jumps = 2  # this is the number of knots at the coarse grid corresponding to slowly changing values

    # get the times of all jumps
    a = [0] + [i+1 for i, x in enumerate(switchpoints_roi) if x] + [len(ROI)]  # get indeces of all the switchpoints, add t0 and tend
    # remove consecutive numbers from the list
    b = []
    for i in range(len(a)):
        if len(b) == 0:  # if the list is empty, we add first item from 'a' (In our example, it'll be 2)
            b.append(a[i])
        else:
            if a[i] > a[i - 1] + 1:  # for every value of a, we compare the last digit from list b
                b.append(a[i])
    jump_indeces = b.copy()
    abs_distance_lists = [[(num - index) for num in range(len(ROI) + 1)] for index in
                          jump_indeces]  # compute absolute distance between each time and time of jump
    min_pos_distances = [min(filter(lambda x: x >= 0, lst)) for lst in zip(*abs_distance_lists)]
    max_neg_distances = [max(filter(lambda x: x <= 0, lst)) for lst in zip(*abs_distance_lists)]
    first_jump_index = np.where(np.array(min_pos_distances) == 0)[0][1]
    min_pos_distances[:first_jump_index] = [np.inf] * len(min_pos_distances[:first_jump_index])
    last_jump_index = np.where(np.array(max_neg_distances) == 0)[0][-2]
    max_neg_distances[last_jump_index:] = [-np.inf] * len(max_neg_distances[last_jump_index:])
    knots_after_jump = [((x <= 2) and (x % 1 == 0)) or ((x <= nPoints_closest) and (x % nPoints_between_closest == 0)) or (
                (nPoints_closest <= x <= nPoints_around_jump) and (x % step_between_knots == 0)) for
        x in min_pos_distances]  # create a knot sequence that has higher density of knots after each jump
    # close_knots_duplicates = [(x <= 1) for x in min_pos_distances]
    # knots_before_jump = [((x >= -nPoints_closest) and (x % (nPoints_closest + 1) == 0)) for x in
    #                      max_neg_distances]  # list on knots befor each jump - use this form if you don't want fine grid before the jump
    knots_before_jump = [(x >= -1) for x in max_neg_distances]  # list on knots before each jump - add a fine grid
    knots_jump = [a or b for a, b in zip(knots_after_jump, knots_before_jump)]
    # add t0 and t_end as a single point in the end
    knots_jump[0] = True
    knots_jump[-1] = True  # logical sum for two boolean lists
    # to do this we then need to add additional coarse grid of knots between two jumps:
    knot_times = [i + ROI_start for i, x in enumerate(knots_jump) if x]
    # close_knots_duplicate_times = [i + ROI_start for i, x in enumerate(close_knots_duplicates) if x]
    # convert to numeric array again
    # add the final time point in case it is not already included - we need this if we are only adding values after steps
    if not np.isin(ROI_end, knot_times):
        knot_times.append(ROI_end)
    knots_all = knot_times.copy() # + close_knots_duplicate_times.copy() # to see if having two splines tied to the close knots will improve precision
    for iKnot, timeKnot in enumerate(knot_times[:-1]):
        # add coarse grid knots between jumps
        if knot_times[iKnot + 1] - timeKnot > step_between_knots:
            # create evenly spaced points and drop start and end - those are already in the grid
            knots_between_jumps = np.rint(
                np.linspace(timeKnot, knot_times[iKnot + 1], num=nPoints_between_jumps + 2)[1:-1]).astype(int)
            # add indeces to the list
            knots_all = knots_all + list(knots_between_jumps)
        # add copies of the closest points to the jump
    knots_all.sort()  # sort list in ascending order - this is done inplace!
    # build the collocation matrix using the defined knot structure
    degree = 3
    fig, ax = plt.subplots()
    outer = [knots_all[0], knots_all[0], knots_all[0], knots_all[-1], knots_all[-1], knots_all[-1]]
    outer_y = []
    knots = np.insert(outer, 3, knots_all)  # create knots for which we want to build splines
    coeffs = np.zeros(len(knots) - degree - 1)  # number of splines will depend on the knot order
    spl_ones = BSpline(knots, np.ones_like(coeffs), degree)
    tau = np.arange(knots[0], knots[-1])
    splinest = [None] * len(coeffs)  # the grid of indtividual splines is required to generate a collocation matrix
    for i in range(len(coeffs)):
        tau_current = np.arange(knots[i], knots[i + 4])
        coeffs[i] = 1
        splinest[i] = BSpline(knots, coeffs.copy(), degree, extrapolate=False)  # create a spline that only has one non-zero coeff
        coeffs[i] = 0
    collocation = collocm(splinest, tau)  # create a collocation matrix for that interval
    ####################################################################################################################
    ## Classes to run optimisation in pints
    nBsplineCoeffs = len(coeffs)  # this to be used in params method of class ForwardModel
    print('Number of B-spline coeffs: ' + str(nBsplineCoeffs))
    nOutputs = 3
    # define a class that outputs only b-spline surface features
    class bsplineOutput(pints.ForwardModel):
        # this model outputs the discrepancy to be used in a rectangle quadrature scheme
        def simulate(self, parameters, times):
            # given times and return the simulated values
            tck = (knots, parameters, degree)
            dot_ = sp.interpolate.splev(times, tck, der=1)
            fun_ = sp.interpolate.splev(times, tck, der=0)
            # the RHS must be put into an array
            rhs = ion_channel_model_one_state(times, fun_, Thetas_ODE)
            return np.array([fun_,dot_,rhs]).T

        def n_parameters(self):
            # Return the dimension of the parameter vector
            return nBsplineCoeffs

        def n_outputs(self):
            # Return the dimension of the output vector
            return nOutputs

    # define an error w.r.t B-spline parameters that assumes that it knows ODE parameters
    class InnerCriterion(pints.ProblemErrorMeasure):
        # do I need to redefine custom init or can just drop this part?
        def __init__(self, problem, weights=None):
            super(InnerCriterion, self).__init__(problem)
            if weights is None:
                weights = [1] * self._n_outputs
            elif self._n_outputs != len(weights):
                raise ValueError(
                    'Number of weights must match number of problem outputs.')
            # Check weights
            self._weights = np.asarray([float(w) for w in weights])
        # this function is the function of beta - bspline parameters
        def __call__(self, betas):
            # evaluate the integral at the value of B-spline coefficients
            model_output = self._problem.evaluate(betas)   # the output of the model with be an array of size nTimes x nOutputs
            x, x_dot, rhs = np.split(model_output, 3, axis=1) # we split the array into states, state derivs, and RHSs
            # compute the data fit
            volts_for_model = self._values[:,1] # we need to make sure that voltage is read at the times within ROI so we pass it in as part of values
            d_y = g * x[:,0] * self._values[:,2] * (volts_for_model - EK) - self._values[:,0]
            data_fit_cost = np.transpose(d_y) @ d_y
            # compute the gradient matching cost
            d_deriv = (x_dot[:] - rhs[:]) ** 2
            integral_quad = sp.integrate.simpson(y=d_deriv, even='avg',axis=0)
            gradient_match_cost = np.sum(integral_quad, axis=0)
            # not the most elegant implementation because it just grabs global lambda
            return data_fit_cost + lambd * gradient_match_cost

    # define a class that outputs only b-spline surface features
    nThetas = len(theta_true)
    class ODEOutput(pints.ForwardModel):
        # this model outputs the discrepancy to be used in a rectangle quadrature scheme
        def simulate(self, parameters, times):
            # given times and return the simulated values
            coeffs = Betas_BSPL
            tck = (knots, coeffs, degree)
            fun_ = sp.interpolate.splev(times, tck, der=0)
            dot_ = sp.interpolate.splev(times, tck, der=1)
            return np.array([fun_,dot_]).T

        def n_parameters(self):
            # Return the dimension of the parameter vector
            return nThetas

        def n_outputs(self):
            # Return the dimension of the output vector
            return 2

    # define an error w.r.t ODE parameters that assumes that it knows B-spline parameters - simply data fit
    class OuterCriterion(pints.ProblemErrorMeasure):
        # do I need to redefine custom init or can just drop this part?
        def __init__(self, problem, weights=None):
            super(OuterCriterion, self).__init__(problem)
            if weights is None:
                weights = [1] * self._n_outputs
            elif self._n_outputs != len(weights):
                raise ValueError(
                    'Number of weights must match number of problem outputs.')
            # Check weights
            self._weights = np.asarray([float(w) for w in weights])
        # this function is the function of theta - ODE parameters
        def __call__(self, thetas):
            # evaluate the integral at the value of ODE parameters
            model_output = self._problem.evaluate(thetas)   # the output of the model with be an array of size nTimes x nOutputs
            x, x_dot = np.split(model_output, 2, axis=1)
            # compute the data fit
            d_y = g * x[:,0] * state_known * (self._values[:,1] - EK) - self._values[:,0] # this part depends on theta_g
            data_fit_cost = np.transpose(d_y) @ d_y
            return data_fit_cost
    ####################################################################################################################
    ## Create objects for the optimisation
    lambd = 1 # 0.3 # 0 # 1
    # set initial values and boundaries
    if inLogScale:
        # theta in log scale
        init_thetas = -5 * np.ones(nThetas)
        sigma0_thetas = 3 * np.ones(nThetas)
        boundaries_thetas = pints.RectangularBoundaries(-12 * np.ones_like(init_thetas),
                                                        -0.5 * np.ones_like(init_thetas))
    else:
        # theta in decimal scale
        init_thetas = 0.001 * np.ones(nThetas)
        sigma0_thetas = 0.0005 * np.ones(nThetas)
        boundaries_thetas = pints.RectangularBoundaries(np.zeros_like(init_thetas), np.ones_like(init_thetas))
    # outer optimisation settings
    ### BEAR IN MIND THAT OUTER OPTIMISATION is conducted on the entire time-series
    model_ode = ODEOutput()
    ## create the problem of comparing the modelled current with measured current
    voltage = V(times_roi)  # must read voltage at the correct times to match the output
    current_true = observation(times_roi, solution.sol(times_roi), theta_true)
    values_to_match_output_ode = np.transpose(np.array([current_true, voltage]))
    # ^ we actually only need first two columns in this array but pints wants to have the same number of values and outputs
    problem_outer = pints.MultiOutputProblem(model=model_ode, times=times_roi,
                                             values=values_to_match_output_ode)
    ## associate the cost with it
    error_outer = OuterCriterion(problem=problem_outer)


    init_betas = 0.5 * np.ones(nBsplineCoeffs) # initial values of B-spline coefficients
    sigma0_betas = 0.2 * np.ones(nBsplineCoeffs)
    tic = tm.time()
    model_bsplines = bsplineOutput()
    ## create the problem of comparing the modelled current with measured current
    voltage = V(times_roi) # must read voltage at the correct times to match the output
    values_to_match_output_dims = np.transpose(np.array([current_true, voltage, state_known]))
    #^ we actually only need first two columns in this array but pints wants to have the same number of values and outputs
    problem_inner = pints.MultiOutputProblem(model=model_bsplines, times=times_roi, values=values_to_match_output_dims)
     ## associate the cost with it
    error_inner  = InnerCriterion(problem=problem_inner)
    ##  define boundaries for the inner optimisation
    boundaries_betas = pints.RectangularBoundaries(np.zeros_like(init_betas), np.ones_like(init_betas))
    ## define boundaries for the outer optimisation
    ####################################################################################################################
    # fit B-spline coefficients to the hidden state directly
    coeffs_ls = np.dot((np.dot(np.linalg.pinv(np.dot(collocation, collocation.T)), collocation)), state_hidden_true.T)
    Betas_BSPL = coeffs_ls[:,0]
    Thetas_ODE = theta_true.copy()
    InnerCost_true = error_inner(Betas_BSPL)
    OuterCost_true = error_outer(Thetas_ODE)
    Betas_BSPL_fit_to_true_states = Betas_BSPL.copy()
    ## get inner cirterion at true ODE param values assuming Betas are unkown
    optimiser_inner = pints.OptimisationController(error_inner, x0=init_betas, sigma0=sigma0_betas, boundaries=boundaries_betas,method=pints.CMAES)
    optimiser_inner.set_max_iterations(30000)
    optimiser_inner.set_max_unchanged_iterations(iterations=50, threshold=1e-6)
    optimiser_inner.set_parallel(False)
    optimiser_inner.set_log_to_screen(True)
    Betas_BSPL_given_true_theta, InnerCost_given_true_theta = optimiser_inner.run()
    Betas_BSPL = Betas_BSPL_given_true_theta.copy()
    OuterCost_given_true_theta = error_outer(Thetas_ODE)

    model_output_fit_to_state = model_bsplines.simulate(Betas_BSPL_fit_to_true_states,times_roi)
    state_direct, state_deriv_direct, rhs_direct = np.split(model_output_fit_to_state, 3, axis=1)
    model_output_fit_at_truth = model_bsplines.simulate(Betas_BSPL_given_true_theta, times_roi)
    state_at_truth, state_deriv_at_truth, rhs_truth = np.split(model_output_fit_at_truth, 3, axis=1)
    current_model_direct = g * state_direct[:,0] * state_known * (voltage - EK)
    current_model_at_truth = g * state_at_truth[:, 0] * state_known * (voltage - EK)
    fig, axes = plt.subplots(3,1,figsize=(12,8),sharex=True)
    y_labels = ['I', '$\dot{' + state_name +'}$', '$'+state_name+'$']
    axes[0].plot(times_roi,current_true, '-k', label=r'Current true',linewidth=2,alpha=0.7)
    axes[0].plot(times_roi,current_model_direct, '--r', label=r'Fit to state directly')
    axes[0].plot(times_roi, current_model_at_truth, '--b', label=r'Optimised given true $\theta$')
    axes[1].plot(times_roi[:],rhs_direct[:], '-k', label='RHS fit directly',linewidth=2,alpha=0.7)
    axes[1].plot(times_roi[:],state_deriv_direct[:], '--r', label=r'B-spline derivative fit directly')
    axes[1].plot(times_roi[:], rhs_truth[:], '-m', label=r'RHS given true $\theta$',linewidth=2,alpha=0.7)
    axes[1].plot(times_roi[:], state_deriv_at_truth[:], '--b', label=r'B-spline derivative given true $\theta$')
    axes[2].plot(times_roi, state_hidden_true[0,:], '-k', label=r'$'+state_name+'$ true',linewidth=2,alpha=0.7)
    axes[2].plot(times_roi, state_direct[:, 0], '--r', label=r'B-spline approximation direct fit')
    axes[2].plot(times_roi, state_at_truth[:, 0], '--b', label=r'B-spline approximation given true $\theta$')
    for iAx, ax in enumerate(axes.flatten()):
        # ax.set_xlim([3380,3420])
        ax.set_ylabel(y_labels[iAx],fontsize=12)
        ax.legend(fontsize=12, loc='upper left')
    ax.set_xlabel('time,ms', fontsize=12)
    plt.tight_layout(pad=0.3)
    # plt.ioff()
    plt.savefig('Figures/cost_terms_at_truth_one_state.png',dpi=600)
    ####################################################################################################################
    # ## compute the costs for the parameters
    sigma = 1.5
    from itertools import combinations
    keys = ['theta_{'+str(index)+'}' for index in np.arange(len(theta_true))]
    explore_costs = dict.fromkeys(keys)
    key_counter = 0
    # check all values within range (mu, mu+3sigma)
    for iTheta,theta in enumerate(theta_true):
        print('iTheta = ' + str(iTheta) + ', theta = ' + str(theta))
        Thetas_ODE = theta_true.copy()
        range_theta_plus = np.linspace(theta, theta + 3*sigma, 50)
        inner_cost_plus = []
        outer_cost_plus = []
        evaluations_plus = []
        evaluations_minus = []
        for iSample, theta_changed in enumerate(range_theta_plus):
            Thetas_ODE[iTheta] = theta_changed.copy()
            # solve ODE
            tic = tm.time()
            optimiser_inner = pints.OptimisationController(error_inner, x0=init_betas, sigma0=sigma0_betas,
                                                           boundaries=boundaries_betas, method=pints.CMAES)
            optimiser_inner.set_max_iterations(30000)
            optimiser_inner.set_max_unchanged_iterations(iterations=50, threshold=1e-6)
            optimiser_inner.set_parallel(False)
            optimiser_inner.set_log_to_screen(False)
            Betas_BSPL, InnerCost_given_theta = optimiser_inner.run()
            nEvals = optimiser_inner._evaluations
            # compute outer cost
            OuterCost_given_theta = error_outer(Thetas_ODE)
            toc = tm.time()
            print(str(iSample) + '-th value checked. Number of evaluations: '+ str(nEvals) + ' Elapsed time: ' + str(toc-tic) + 's.')
            # store all cost values in lists for this particular parameter
            inner_cost_plus.append(InnerCost_given_theta)
            outer_cost_plus.append(OuterCost_given_theta)
            evaluations_plus.append(nEvals)
            # assign found values at this point as inital values for the neighbouring point
            init_betas = Betas_BSPL.copy()
            # if it is the tru value of the parameter, save inital betas to be used in (mu-3sigma,mu) interval
            if iSample==0:
                init_betas_at_truth = Betas_BSPL.copy()
        # check all values within range (mu-3sigma, mu)
        Thetas_ODE = theta_true.copy()
        range_theta_minus = np.linspace(theta, theta - 3 * sigma, 50)
        inner_cost_minus = []
        outer_cost_minus = []
        init_betas = init_betas_at_truth.copy()
        for iSample, theta_changed in enumerate(range_theta_minus[1:]): #skip the minima as we have already evaluated it
            Thetas_ODE[iTheta] = theta_changed.copy()
            # solve ODE
            tic = tm.time()
            optimiser_inner = pints.OptimisationController(error_inner, x0=init_betas, sigma0=sigma0_betas,
                                                           boundaries=boundaries_betas, method=pints.CMAES)
            optimiser_inner.set_max_iterations(30000)
            optimiser_inner.set_max_unchanged_iterations(iterations=50, threshold=1e-6)
            optimiser_inner.set_parallel(False)
            optimiser_inner.set_log_to_screen(False)
            Betas_BSPL, InnerCost_given_theta = optimiser_inner.run()
            nEvals = optimiser_inner._evaluations
            # compute outer cost
            OuterCost_given_theta = error_outer(Thetas_ODE)
            toc = tm.time()
            print(str(iSample) + '-th value checked. Number of evaluations: '+ str(nEvals) + ' Elapsed time: ' + str(toc-tic) + 's.')
            # store all cost values in lists for this particular parameter
            inner_cost_minus.append(InnerCost_given_theta)
            outer_cost_minus.append(OuterCost_given_theta)
            evaluations_minus.append(nEvals)
            # assign found values at this point
            init_betas = Betas_BSPL.copy()

        # combine lists to get the interval (mu-3sigma, mu+3sigma)
        inner_cost_minus.reverse()
        outer_cost_minus.reverse()
        evaluations_minus.reverse()
        range_theta = list(np.flip(range_theta_minus[1:])) + list(range_theta_plus)
        inner_cost_plot = inner_cost_minus + inner_cost_plus
        outer_cost_plot = outer_cost_minus + outer_cost_plus
        evaluations_total = evaluations_minus + evaluations_plus
        explore_costs[keys[key_counter]] = [range_theta, inner_cost_plot, outer_cost_plot, evaluations_total]
        key_counter += 1
    ####################################################################################################################
    # # save the exploration results
    # create metadata dictionary:
    metadata = {'times': times_roi, 'lambda': lambd, 'state_name': state_name, 'state_true': state_hidden_true, 'state_known': state_known,
                'knots': knots, 'truth': theta_true, 'param_names': param_names, 'log_scaled': inLogScale}
    # write into all into file
    with open("Pickles/explored_parameter_space_with_init_changing_"+state_name+".pkl", "wb") as output_file:
        pkl.dump([explore_costs, metadata], output_file)
    ###################################################################################################################
    nColumns = len(keys)
    fig, axes = plt.subplots(2,nColumns)
    for iKey, key in enumerate(keys):
        axes[0,iKey].semilogy(explore_costs[key][0],explore_costs[key][1],label='Inner cost')
        axes[0,iKey].semilogy(theta_true[iKey], InnerCost_true, lw=0, color='blue', marker='s', label='Direct fit at truth')
        axes[0,iKey].semilogy(theta_true[iKey], InnerCost_given_true_theta,lw=0, color='magenta', marker='o', label='Collocation at truth')
        ind_min = np.argmin(explore_costs[key][1])
        axes[0, iKey].semilogy(explore_costs[key][0][ind_min], explore_costs[key][1][ind_min],lw=0, color='black', marker='.', label='Empirical min')
        axes[0,iKey].set_xlabel(r'$\theta_{' + str(iKey+1) + '} = log(' + param_names[iKey] +')$')
        axes[0,iKey].set_ylabel(r'$J(C \mid \theta_{' + str(iKey+1) + '}, \mathbf{y})$')
        axes[0,iKey].legend(loc='best')
        axes[1,iKey].semilogy(explore_costs[key][0], explore_costs[key][2], label='Outer cost')
        axes[1,iKey].semilogy(theta_true[iKey], OuterCost_true, lw=0, color='blue', marker='s', label='Direct fit at truth')
        axes[1,iKey].semilogy(theta_true[iKey], OuterCost_given_true_theta,lw=0, color='magenta', marker='o', label='Collocation at truth')
        ind_min = np.argmin(explore_costs[key][2])
        axes[1,iKey].semilogy(explore_costs[key][0][ind_min], explore_costs[key][2][ind_min],lw=0, color='black', marker='.', label='Empirical min')
        axes[1,iKey].set_xlabel(r'$\theta_{' + str(iKey+1) + '} = log(' + param_names[iKey] + ')$')
        axes[1,iKey].set_ylabel(r'$G_{y}(\theta_{' + str(iKey+1) + '} \mid \mathbf{y})$')
        axes[1,iKey].legend(loc='best')
    plt.tight_layout(pad=0.3)
    plt.savefig('Figures/costs_projection_semilogy_changing_inits_'+state_name+'.png',dpi=400)

    # #     get the diagonal ranges for all pairs of parameters using unique combinations
    # for iPair,pair in enumerate(combinations(range(len(theta_true)),2)):
    #     Thetas_ODE = theta_true
    #     range_theta = np.linspace(-8, 0, 101)
    #     inner_cost_plot = []
    #     outer_cost_plot = []
    #     for iSample, theta_changed in enumerate(range_theta):
    #         Thetas_ODE[pair[0]] = Thetas_ODE[pair[1]] = theta_changed.copy()
    #         # solve ODE
    #         tic = tm.time()
    #         optimiser_inner = pints.OptimisationController(error_inner, x0=init_betas, sigma0=sigma0_betas,
    #                                                        boundaries=boundaries_betas, method=pints.CMAES)
    #         optimiser_inner.set_max_iterations(30000)
    #         optimiser_inner.set_max_unchanged_iterations(iterations=50, threshold=1e-6)
    #         optimiser_inner.set_parallel(False)
    #         optimiser_inner.set_log_to_screen(False)
    #         Betas_BSPL, InnerCost_given_theta = optimiser_inner.run()
    #         # copmute outer cost
    #         OuterCost_given_theta = error_outer(Thetas_ODE)
    #         toc = tm.time()
    #         print(str(iSample) + '-th value checked. Elapsed time: ' + str(toc - tic) + 's.')
    #         # store all cost values in lists for this particular parameter
    #         inner_cost_plot.append(InnerCost_given_theta)
    #         outer_cost_plot.append(OuterCost_given_theta)
    #     explore_costs['theta_{'+str(pair[0])+'} = theta_{'+str(pair[1])+'}$'] = [range_theta, inner_cost_plot, outer_cost_plot]
    ####################################################################################################################
    # # # save the exploration results
    # with open("Pickles/explored_parameter_space.pkl", "wb") as output_file:
    #     pkl.dump(explore_costs, output_file)
    ####################################################################################################################
    # take 1: loosely based on ask-tell example from  pints
    convergence_threshold = 1e-5
    iter_for_convergence = 50
    # Create an outer optimisation object
    big_tic = tm.time()
    optimiser_outer = pints.CMAES(x0=init_thetas,sigma0=sigma0_thetas, boundaries=boundaries_thetas)
    optimiser_outer.set_population_size(min(len(Thetas_ODE)*7,30))
    ## Run optimisation
    theta_visited = []
    theta_guessed = []
    f_guessed = []
    theta_best = []
    f_best = []
    InnerCosts_all = []
    OuterCosts_all = []
    for i in range(500):
        # get the next points (multiple locations)
        thetas = optimiser_outer.ask()
        # create the placeholder for cost functions
        OuterCosts = []
        InnerCosts = []
        # for each theta in the sample
        tic = tm.time()
        for theta in thetas:
            # assign the variable that is readable in the class of B-spline evaluation
            Thetas_ODE = theta.copy()
            # fit the b-spline surface given the sampled value of the ODE parameter vector
            # introduce an optimiser every time beacause it does not understand why thre is already an instance of the optimier
            optimiser_inner = pints.OptimisationController(error_inner, x0=init_betas, boundaries=boundaries_betas,
                                                           method=pints.CMAES)
            optimiser_inner.set_max_iterations(30000)
            optimiser_inner.set_max_unchanged_iterations(iterations=50, threshold=1e-8)
            optimiser_inner.set_parallel(False)
            optimiser_inner.set_log_to_screen(False)
            Betas_BSPL, InnerCost = optimiser_inner.run()
            # init_betas = Betas_BSPL # update the init conds for the next itimiser instance
            # evaluate the cost function at the sampled value of ODE parameter vector
            InnerCosts.append(InnerCost)
            OuterCosts.append(error_outer(theta))
            del Thetas_ODE # make sure this is updated
        # feed the evaluated scores into the optimisation object
        optimiser_outer.tell(OuterCosts)
        toc = tm.time()
        print(str(i) + '-th iteration finished. Elapsed time: ' + str(toc-tic) + 's')
        # store all costs in the lists
        InnerCosts_all.append(InnerCosts)
        OuterCosts_all.append(OuterCosts)
        # HOW DO I CHECK CONVERGENCE HERE - for all points of average cost???
        # Store the requested points
        theta_visited.extend(thetas)
        # Store the current guess
        theta_g =np.mean(thetas, axis=0)
        theta_guessed.append(theta_g)
        f_guessed.append(error_outer(theta_g))
        # Store the accompanying score
        # Store the best position and score seen so far
        index_best = OuterCosts.index(min(OuterCosts))
        theta_best.append(thetas[index_best,:])
        f_best.append(OuterCosts[index_best])
        # the most basic convergence condition after running first fifty
        if (i > iter_for_convergence):
            # check how the cost increment changed over the last 10 iterations
            d_cost = np.diff(f_best[-iter_for_convergence:])
            # if all incrementa are below a threshold break the loop
            if all(d<=convergence_threshold for d in d_cost):
                print("No changes in" + str(iter_for_convergence) + "iterations. Terminating")
                break
    # convert lists into arrays
    theta_visited = np.array(theta_visited)
    theta_guessed = np.array(theta_guessed)
    theta_best = np.array(theta_best)
    f_best = np.array(f_best)
    f_guessed = np.array(f_guessed)
    big_toc = tm.time()
    print('Optimisation finished. Elapsed time: ' + str(big_toc-big_tic) + 's')
    ####################################################################################################################
    # try saving the results
    results_to_save = [InnerCosts_all,OuterCosts_all,theta_visited,theta_guessed,theta_best,f_guessed,f_best]
    with open("Pickles/ask_tell_simple_problem_iterations.pkl", "wb") as output_file:
        pkl.dump([results_to_save,metadata], output_file)
    ####################################################################################################################
    # plot evolution of outer costs
    plt.figure(figsize=(10, 6))
    plt.semilogy()
    plt.xlabel('Iteration')
    plt.ylabel('Inner optimisation cost')
    for iter in range(len(f_best)-1):
        plt.scatter(iter*np.ones(len(InnerCosts_all[iter])),InnerCosts_all[iter], c='k',marker='.', alpha=.5, linewidths=0)
    iter += 1
    plt.scatter(iter * np.ones(len(InnerCosts_all[iter])), InnerCosts_all[iter], c='k', marker='.', alpha=.5,
                linewidths=0,label=r'Sample cost min: $J(C \mid \Theta, \bar{\mathbf{y}}) = $'  +"{:.7f}".format(min(InnerCosts_all[iter])) )
    plt.plot(range(iter), np.ones(iter) * InnerCost_true, '-m', linewidth=2.5, alpha=.5, label=r'B-splines fit to true state: $J(C \mid  \mathbf{x}_{true}) = $' +"{:.7f}".format(InnerCost_true))
    plt.plot(range(iter), np.ones(iter) * InnerCost_given_true_theta, '--b', linewidth=2.5, alpha=.5, label=r'Collocation solution: $J(C \mid \Theta_{true}, \bar{\mathbf{y}}) = $'  +"{:.7f}".format(InnerCost_given_true_theta))
    plt.legend(loc='best')
    plt.tight_layout()
    plt.savefig('Figures/inner_cost_ask_tell_one_state.png',dpi=400)

    # plot evolution of inner costs
    plt.figure(figsize=(10, 6))
    plt.semilogy()
    plt.xlabel('Iteration')
    plt.ylabel('Outer optimisation cost')
    for iter in range(len(f_best) - 1):
        plt.scatter(iter * np.ones(len(OuterCosts_all[iter])), OuterCosts_all[iter], c='k', marker='.', alpha=.5,
                    linewidths=0)
    iter += 1
    plt.scatter(iter * np.ones(len(OuterCosts_all[iter])), OuterCosts_all[iter], c='k', marker='.', alpha=.5,linewidths=0, label=r'Sample cost: $H(\Theta \mid \hat{C}, \bar{\mathbf{y}})$')
    plt.plot(range(iter), np.ones(iter) * OuterCost_true, '-m', linewidth=2.5, alpha=.5,label=r'B-splines fit to true state: $H(\Theta \mid  \hat{C}_{direct}, \bar{\mathbf{y}}) = $' + "{:.7f}".format(
                 OuterCost_true))
    plt.plot(range(iter), np.ones(iter) * OuterCost_given_true_theta, '--b', linewidth=2.5, alpha=.5,label=r'Collocation solution: $H(\Theta_{true} \mid  \hat{C}, \bar{\mathbf{y}}) = $' + "{:.7f}".format(
                 OuterCost_given_true_theta))
    plt.plot(f_best,'-b',linewidth=1.5,label=r'Best cost:$H(\Theta_{true} \mid  \hat{C}, \bar{\mathbf{y}}) = $' + "{:.7f}".format(f_best[-1]))
    plt.legend(loc='best')
    plt.tight_layout()
    plt.savefig('Figures/outer_cost_ask_tell_one_state.png',dpi=400)

    # plot parameter values after search was done on decimal scale
    fig, axes = plt.subplots(len(theta_true), 1, figsize=(5*len(theta_true), 8), sharex=True)
    n_walkers = int(theta_visited.shape[0] / len(theta_best))
    for iAx, ax in enumerate(axes.flatten()):
        for iter in range(len(theta_best)):
            x_visited_iter = theta_visited[iter*n_walkers:(iter+1)*n_walkers,iAx]
            ax.scatter(iter*np.ones(len(x_visited_iter)),x_visited_iter,c='k',marker='.',alpha=.2,linewidth=0)
        ax.plot(range(iter+1),np.ones(iter+1)*theta_true[iAx], '-m', linewidth=2.5,alpha=.5, label=r"true: $\theta_{"+str(iAx+1)+"} = $" +"{:.4f}".format(theta_true[iAx]))
        ax.plot(theta_guessed[:,iAx],'--r',linewidth=1.5,label=r"guessed: $\theta_{"+str(iAx+1)+"} = $" +"{:.4f}".format(theta_guessed[-1,iAx]))
        ax.plot(theta_best[:,iAx],'-b',linewidth=1.5,label=r"best: $\theta_{"+str(iAx+1)+"} = $" +"{:.4f}".format(theta_best[-1,iAx]))
        ax.set_ylabel(r'$\theta_{'+str(iAx+1)+'}$')
        ax.legend(loc='best')
    plt.tight_layout()
    plt.savefig('Figures/ODE_params_one_state.png',dpi=400)

    # plot parameter values converting from log scale to decimal
    fig, axes = plt.subplots(len(theta_true), 1, figsize=(5*len(theta_true), 8), sharex=True)
    n_walkers = int(theta_visited.shape[0] / len(theta_best))
    for iAx, ax in enumerate(axes.flatten()):
        for iter in range(len(theta_best)):
            x_visited_iter = theta_visited[iter*n_walkers:(iter+1)*n_walkers,iAx]
            ax.scatter(iter*np.ones(len(x_visited_iter)),np.exp(x_visited_iter),c='k',marker='.',alpha=.2,linewidth=0)
        ax.plot(range(iter+1),np.ones(iter+1)*np.exp(theta_true[iAx]), '-m', linewidth=2.5,alpha=.5, label="true: $a_{"+str(iAx+1)+"} = $" +"{:.4f}".format(np.exp(theta_true[iAx])))
        ax.plot(np.exp(theta_guessed[:,iAx]),'--r',linewidth=1.5,label="guessed: $a_{"+str(iAx+1)+"} = $" +"{:.4f}".format(np.exp(theta_guessed[-1,iAx])))
        ax.plot(np.exp(theta_best[:,iAx]),'-b',linewidth=1.5,label="best: $a_{"+str(iAx+1)+"} = $" +"{:.4f}".format(np.exp(theta_best[-1,iAx])))
        ax.set_ylabel('$a_{'+str(iAx+1)+'}$')
        ax.set_yscale('log')
        ax.legend(loc='best')
    ax.set_ylabel('Iteration')
    plt.tight_layout()
    plt.savefig('Figures/ODE_params_one_state_log_scale.png',dpi=400)


    # plot model output
    current_true = observation(times_roi, x_ar, p_true)
    Thetas_ODE = theta_best[-1,:]
    optimiser_inner = pints.OptimisationController(error_inner, x0=init_betas, boundaries=boundaries_betas,
                                                   method=pints.CMAES)
    optimiser_inner.set_max_iterations(30000)
    optimiser_inner.set_max_unchanged_iterations(iterations=50, threshold=1e-6)
    optimiser_inner.set_parallel(False)
    optimiser_inner.set_log_to_screen(False)
    Betas_BSPL, BSPL_cost = optimiser_inner.run()

    # get model output and plot all curves
    opt_model_output = model_bsplines.simulate(Betas_BSPL,times_roi)
    state, state_deriv, rhs = np.split(opt_model_output, 3, axis=1)
    *ps, g = p_true[:9]
    current_model = g * state[:,0] * state_known * (voltage - EK)
    fig, axes = plt.subplots(3,1,figsize=(14,9),sharex=True)
    axes[0].plot(times_roi,current_true, '-k', label='Current true')
    axes[0].plot(times_roi,current_model, '--r', label='Optimised model output')
    axes[1].plot(times_roi,rhs, '-k', label='RHS at collocation solution')
    axes[1].plot(times_roi,state_deriv, '--r', label='B-spline derivative')
    axes[2].plot(times_roi, state_hidden_true[0,:], '-k', label='$'+ state_name +'$ true')
    axes[2].plot(times_roi, state[:, 0], '--r', label='Collocation solution')
    for iAx, ax in enumerate(axes.flatten()):
        ax.legend(fontsize=12, loc='best')
        ax.set_ylabel(y_labels[iAx],fontsize=12)
    ax.set_xlabel('time,ms', fontsize=12)
    plt.tight_layout(pad=0.3)
    # plt.ioff()
    plt.savefig('Figures/cost_terms_ask_tell_one_state.png',dpi=400)
    ####################################################################################################################
    print('pause here')