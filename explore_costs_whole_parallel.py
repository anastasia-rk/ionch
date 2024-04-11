# imports
import numpy as np

from setup import *
import pints
import pickle as pkl
import traceback
import multiprocessing as mp
from itertools import repeat
matplotlib.use('AGG')
plt.ioff()

# definitions
def hh_model(t, x, theta):
    a, r = x[:2]
    *p, g = theta[:9]
    v = V(t)
    k1 = p[0] * np.exp(p[1] * v)
    k2 = p[2] * np.exp(-p[3] * v)
    k3 = p[4] * np.exp(p[5] * v)
    k4 = p[6] * np.exp(-p[7] * v)
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
    return volts_intepolated((t)/ 1000)

### Only consider a -- all params in log scale
def ode_a_only(t, x, theta):
    # call the model with a smaller number of unknown parameters and one state known
    a = x
    v = V(t)
    k1 =  np.exp(theta[0] + np.exp(theta[1]) * v)
    k2 =  np.exp(theta[2] -np.exp(theta[3]) * v)
    a_inf = k1 / (k1 + k2)
    tau_a = 1 / (k1 + k2)
    da = (a_inf - a) / tau_a
    return da

### Only  consider r -- log space on a parameters
def ode_r_only(t, x, theta):
    # call the model with a smaller number of unknown parameters and one state known
    r = x
    v = V(t)
    k3 =  np.exp(theta[0] + np.exp(theta[1]) * v)
    k4 =  np.exp(theta[2] - np.exp(theta[3]) * v)
    r_inf = k4 / (k3 + k4)
    tau_r = 1 / (k3 + k4)
    dr = (r_inf - r) / tau_r
    return dr

# define inner optimisation as a function to parallelise the CMA-ES
def inner_optimisation(theta, segment, input_segment, output_segment, knots, state_known_segment, init_betas):
    # assign the variable that is readable in the class of B-spline evaluation
    global Thetas_ODE # declrae the global variable to be used in classess across all functions
    Thetas_ODE = theta.copy()
    # fit the b-spline surface given the sampled value of the ODE parameter vector
    optimisationFailed = False
    # create an optimisaton object
    sigma0_betas = 0.2 * np.ones_like(init_betas)  # inital spread of values
    state_fitted = {key: [] for key in hidden_state_names}
    # run the optimisation
    tic_run = tm.time()
    try:
        optimiser_inner = pints.OptimisationController(error_inner, x0=init_betas, sigma0=sigma0_betas,
                                                       boundaries=boundaries_betas, method=pints.CMAES)
        optimiser_inner.set_max_iterations(100000)
        optimiser_inner.set_max_unchanged_iterations(iterations=50, threshold=1e-8)
        optimiser_inner.set_parallel(False)
        optimiser_inner.set_log_to_screen(False)
        # run the optimisation
        betas_run, inner_costs_run = optimiser_inner.run()
        evaluations_run = optimiser_inner._evaluations
    except Exception:
        traceback.print_exc()
        print('Error encountered during optimisation.')
        optimisationFailed = True
        return (np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, optimisationFailed)  # return dummy values
    else:
        # check collocation solution against truth
        model_output = model_bsplines_test.simulate(betas_run, segment)
        state_at_estimate, deriv_at_estimate, rhs_at_estimate = np.split(model_output, 3, axis=1)
        current_model_at_estimatete = g * state_at_estimate[:, 0] * state_known_segment * (
                input_segment - EK)
        dy = (current_model_at_estimatete - output_segment)
        d_deriv = np.square(np.subtract(deriv_at_estimate, rhs_at_estimate))
        integral_quad = sp.integrate.simpson(y=d_deriv, even='avg', axis=0)
        grad_costs_run = np.sum(integral_quad, axis=0)
        outer_costs_run = dy @ np.transpose(dy)
        for iState, stateName in enumerate(hidden_state_names):
            state_fitted[stateName] += list(state_at_estimate[:, iState])
    toc_run = tm.time()
    runtime_run = toc_run - tic_run
    # print results of the run
    print('Run completed in ' + str(runtime_run) + ' seconds.')
    print('Inner cost: ' + str(inner_costs_run))
    print('Outer cost: ' + str(outer_costs_run))
    print('Gradient cost: ' + str(grad_costs_run))
    result = (betas_run, inner_costs_run, outer_costs_run, grad_costs_run, evaluations_run, runtime_run, state_fitted, optimisationFailed)
    return result

def plot_segment(iRoi, roi,true_output_roi,model_output_roi, model_state_roi ,state_true, state_name):
    state_, deriv_, rhs_ = np.split(model_state_roi, 3, axis=1)
    fig, axes = plt.subplot_mosaic([['a)'], ['b)'], ['c)']], layout='constrained')
    y_labels = ['I', '$\dot{' + state_name + '}$', '$' + state_name + '$']
    axes['a)'].plot(roi, true_output_roi, '-k', label=r'Current true', linewidth=2, alpha=0.7)
    axes['a)'].plot(roi, model_output_roi, '--b', label=r'Optimised given true $\theta$')
    axes['b)'].plot(roi, rhs_[:, 0], '-m', label=r'$\dot{' + state_name + '}$ given true $\theta$',
                    linewidth=2,
                    alpha=0.7)
    axes['b)'].plot(roi, deriv_[:, 0], '--b',
                    label=r'B-spline derivative given true $\theta$')
    axes['c)'].plot(roi, state_true, '-k', label=r'$' + state_name + '$ true', linewidth=2,
                    alpha=0.7)
    axes['c)'].plot(roi, state_[:, 0], '--b', label=r'B-spline approximation given true $\theta$')
    iAx = 0
    for _, ax in axes.items():
        ax.set_ylabel(y_labels[iAx], fontsize=12)
        ax.legend(fontsize=12, loc='upper left')
        iAx += 1
    # plt.tight_layout(pad=0.3)
    plt.savefig('Figures/cost_terms_at_truth_one_state_segment_' + str(iRoi) + '.png', dpi=400)

# main
if __name__ == '__main__':
    #  load the voltage data:
    volts = np.genfromtxt("./protocol-staircaseramp.csv", skip_header=1, dtype=float, delimiter=',')
    #  check when the voltage jumps
    # read the times and valued of voltage clamp
    volt_times, volts = np.genfromtxt("./protocol-staircaseramp.csv", skip_header=1, dtype=float, delimiter=',').T
    # interpolate with smaller time step (milliseconds)
    volts_intepolated = sp.interpolate.interp1d(volt_times, volts, kind='previous')

    ## define the time interval on which the fitting will be done
    tlim = [3500, 10300]
    times = np.linspace(*tlim, tlim[-1]-tlim[0],endpoint=False)
    volts_new = V(times)
    ## Generate the synthetic data
    # parameter values for the model
    EK = -80
    thetas_true = [2.26e-4, 0.0699, 3.45e-5, 0.05462, 0.0873, 8.91e-3, 5.15e-3, 0.03158, 0.1524]
    *ps, g = thetas_true[:9]
    # initialise and solve ODE
    x0 = [0, 1]
    state_names = ['a','r']
    # solve initial value problem
    solution = sp.integrate.solve_ivp(hh_model, [0,tlim[-1]], x0, args=[thetas_true], dense_output=True,method='LSODA',rtol=1e-8,atol=1e-8)
    x_ar = solution.sol(times)
    current_true = observation(times, x_ar, thetas_true)

    # ## single state model
    # # use a as unknown state
    # theta_true = [np.log(2.26e-4), np.log(0.0699), np.log(3.45e-5), np.log(0.05462)]
    # inLogScale = True
    # param_names = ['p_1','p_2','p_3','p_4']
    # a0 = [0]
    # ion_channel_model_one_state = ode_a_only
    # solution_one_state = sp.integrate.solve_ivp(ion_channel_model_one_state, [0,tlim[-1]], a0, args=[theta_true], dense_output=True, method='LSODA',
    #                                   rtol=1e-8, atol=1e-8)
    # state_known_index = state_names.index('r')  # assume that we know r
    # state_known = x_ar[state_known_index, :]
    # state_name = hidden_state_names = 'a'

    ## use r as unknown state
    theta_true = [np.log(0.0873), np.log(8.91e-3), np.log(5.15e-3), np.log(0.03158)]
    inLogScale = True
    param_names = ['p_5','p_6','p_7','p_8']
    r0 = [1]
    ion_channel_model_one_state = ode_r_only
    solution_one_state = sp.integrate.solve_ivp(ion_channel_model_one_state, [0,tlim[-1]], r0, args=[theta_true], dense_output=True,
                                        method='LSODA',
                                        rtol=1e-8, atol=1e-10)
    state_known_index = state_names.index('a')  # assume that we know a
    state_known = x_ar[state_known_index,:]
    state_name = hidden_state_names = 'r'
    ################################################################################################################
    ## store true hidden state
    state_hidden_true = x_ar[state_names.index(state_name), :]
    rhs_true = ion_channel_model_one_state(times, state_hidden_true, theta_true)
    ################################################################################################################
    ## B-spline representation setup
    # set times of jumps and a B-spline knot sequence
    nPoints_closest = 6  # the number of points from each jump where knots are placed at the finest grid
    nPoints_between_closest = 2  # step between knots at the finest grid
    nPoints_around_jump = 80  # the time period from jump on which we place medium grid
    step_between_knots = 16  # this is the step between knots around the jump in the medium grid
    nPoints_between_jumps = 2  # this is the number of knots at the coarse grid corresponding to slowly changing values
    ## find switchpoints
    d2v_dt2 = np.diff(volts_new, n=2)
    dv_dt = np.diff(volts_new)
    der1_nonzero = np.abs(dv_dt) > 1e-6
    der2_nonzero = np.abs(d2v_dt2) > 1e-6
    switchpoints = [a and b for a, b in zip(der1_nonzero, der2_nonzero)]
    ####################################################################################################################
    # get the times of all jumps
    a = [0] + [i for i, x in enumerate(switchpoints) if x] + [len(times)-1]  # get indeces of all the switchpoints, add t0 and tend
    # remove consecutive numbers from the list
    b = []
    for i in range(len(a)):
        if len(b) == 0:  # if the list is empty, we add first item from 'a' (In our example, it'll be 2)
            b.append(a[i])
        else:
            if a[i] > a[i - 1] + 1:  # for every value of a, we compare the last digit from list b
                b.append(a[i])
    jump_indeces = b.copy()
    knots_roi = []
    for iJump, jump in enumerate(jump_indeces[:-1]):  # loop oversegments (nJumps - )
        # define a region of interest - we will need this to preserve the
        # trajectories of states given the full clamp and initial position, while
        ROI_start = jump
        ROI_end = jump_indeces[iJump + 1]  # add one to ensure that t_end equals to t_start of the following segment
        ROI = times[ROI_start:ROI_end]
        ## add colloation points
        abs_distance_lists = [[(num - index) for num in range(ROI_start, ROI_end-1)] for index in
                              [ROI_start, ROI_end]]  # compute absolute distance between each time and time of jump
        min_pos_distances = [min(filter(lambda x: x >= 0, lst)) for lst in zip(*abs_distance_lists)]
        max_neg_distances = [max(filter(lambda x: x <= 0, lst)) for lst in zip(*abs_distance_lists)]
        # create a knot sequence that has higher density of knots after each jump
        knots_after_jump = [((x <= nPoints_closest) and (x % nPoints_between_closest == 0)) or (
                (nPoints_closest < x <= nPoints_around_jump) and (x % step_between_knots == 0)) for
                            x in min_pos_distances]  ##  ((x <= 2) and (x % 1 == 0)) or
        # knots_before_jump = [((x >= -nPoints_closest) and (x % (nPoints_closest + 1) == 0)) for x in
        #                      max_neg_distances]  # list on knots befor each jump - use this form if you don't want fine grid before the jump
        knots_before_jump = [(x >= -1) for x in max_neg_distances]  # list on knots before each jump - add a fine grid
        knots_jump = [a or b for a, b in
                      zip(knots_after_jump, knots_before_jump)]  # logical sum of mininal and maximal distances
        # convert to numeric array again
        knot_indeces = [i + ROI_start for i, x in enumerate(knots_jump) if x]
        if not np.isin(ROI_end, knot_indeces):
            knot_indeces.append(ROI_end)
        indeces_inner = knot_indeces.copy()
        # add additional coarse grid of knots between two jumps:
        for iKnot, timeKnot in enumerate(knot_indeces[:-1]):
            # add coarse grid knots between jumps
            if knot_indeces[iKnot + 1] - timeKnot > step_between_knots:
                # create evenly spaced points and drop start and end - those are already in the grid
                knots_between_jumps = np.rint(
                    np.linspace(timeKnot, knot_indeces[iKnot + 1], num=nPoints_between_jumps + 2)[1:-1]).astype(int)
                # add indeces to the list
                indeces_inner = indeces_inner + list(knots_between_jumps)
            # add copies of the closest points to the jump
        ## end loop over knots
        indeces_inner.sort()  # sort list in ascending order - this is done inplace
        # save knots between jumps
        knots_roi.append(indeces_inner)
    # end of loop over jump indeces, we now need to add additional Boor points on each side
    degree = 3
    indeces_all = [item for sublist in knots_roi for item in sublist]
    used = set()
    indeces_unique = [x for x in indeces_all if x not in used and (used.add(x) or True)]
    indeces_outer = [indeces_unique[0]] * 3 + [indeces_unique[-1]] * 3
    boor_indeces = np.insert(indeces_outer, degree,
                             indeces_unique)
    knots = times[boor_indeces]
    nBsplineCoeffs = len(knots) - degree - 1  # this to be used in params method of class ForwardModel
    print('Number of B-spline coeffs: ' + str(nBsplineCoeffs))
    ####################################################################################################################
    ## create a list of indeces to insert first B-spline coeffs for each segment
    indeces_to_add = [0]
    for iState in range(1, len(hidden_state_names)):
        indeces_to_add.append((len(coeffs) - 1) * iState)
    ## create a list of indeces to drop from the B-spline coeff sets for each segment
    indeces_to_drop = [0]
    for iState in range(1, len(hidden_state_names)):
        indeces_to_drop.append(int(len(coeffs) * iState))
    upper_bound_beta = 0.9999
    nSegments = iJump
    ####################################################################################################################
    ## define pints classes for optimisation
    lambd = 1 #0.000001
    ## Classes to run optimisation in pints
    init_betas = 0.5 * np.ones(nBsplineCoeffs)  # initial values of B-spline coefficients
    sigma0_betas = 0.2 * np.ones(nBsplineCoeffs) # sigma of inital spread of values
    Thetas_ODE = theta_true.copy()
    print('Number of B-spline coeffs per segment: ' + str(nBsplineCoeffs))
    nOutputs = 3
    # define a class that outputs only b-spline surface features - we need it to compute outer cost
    class bsplineOutputTest(pints.ForwardModel):
        # this model outputs the discrepancy to be used in a rectangle quadrature scheme
        def simulate(self, parameters, times):
            # given times and return the simulated values
            tck = (knots, parameters, degree)
            dot_ = sp.interpolate.splev(times, tck, der=1)
            fun_ = sp.interpolate.splev(times, tck, der=0)
            # the RHS must be put into an array
            rhs = ion_channel_model_one_state(times, fun_, Thetas_ODE)
            return np.array([fun_, dot_, rhs]).T

        def n_parameters(self):
            # Return the dimension of the parameter vector
            return nBsplineCoeffs

        def n_outputs(self):
            # Return the dimension of the output vector
            return nOutputs
    #############################################################
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
            model_output = self._problem.evaluate(betas)  # the output of the model with be an array of size nTimes x nOutputs
            x, x_dot, rhs = np.split(model_output, 3, axis=1)  # we split the array into states, state derivs, and RHSs
            # compute the data fit
            volts_for_model = self._values[:, 1]  # we need to make sure that voltage is read at the times within ROI so we pass it in as part of values
            d_y = g * x[:, 0] * self._values[:, 2] * (volts_for_model - EK) - self._values[:, 0]
            data_fit_cost = np.transpose(d_y) @ d_y
            # compute the gradient matching cost
            d_deriv = np.square(np.subtract(x_dot,rhs))
            integral_quad = sp.integrate.simpson(y=d_deriv, even='avg', axis=0) # this computes integrals for all states
            gradient_match_cost = np.sum(integral_quad, axis=0) # we sum the integrals of multiple hidden states
            # not the most elegant implementation because it just grabs global lambda
            return data_fit_cost + lambd * gradient_match_cost
    ####################################################################################################################
    ## set up number of runs and cores
    ncpu = mp.cpu_count()
    nRuns = ncpu
    ncores = 10
    ###############################################################################################################
    ## set up the optimisation
    *ps, g = thetas_true
    model_bsplines = bsplineOutputTest()
    init_betas = 0.5 * np.ones(nBsplineCoeffs)  # initial values of B-spline coefficients
    sigma0_betas = 0.2 * np.ones(nBsplineCoeffs)
    ## create the problem of comparing the modelled current with measured current
    voltage = V(times)  # must read voltage at the correct times to match the output
    output = observation(times, x_ar, thetas_true)
    values_to_match_output_dims = np.transpose(np.array([output, voltage, state_known]))
    # ^ we actually only need first two columns in this array but pints wants to have the same number of values and outputs
    problem_inner = pints.MultiOutputProblem(model=model_bsplines, times=times, values=values_to_match_output_dims)
    ## associate the cost with it
    error_inner = InnerCriterion(problem=problem_inner)
    ##  define boundaries for the inner optimisation
    boundaries_betas = pints.RectangularBoundaries(np.zeros_like(init_betas), np.ones_like(init_betas))
    ###############################################################################################################
    # ## compute the costs for the parameters
    sigma = 1 # std from thhe true value
    nSamples = 15 # number of samples on either side of the true values
    keys = ['theta_{' + str(index) + '}' for index in np.arange(len(theta_true))]
    explore_costs = dict.fromkeys(keys)
    key_counter = 0
    model_bsplines_test = bsplineOutputTest()
    directions = ['positive','negative']
    # check all parameter values within range (mu, mu+3sigma)
    for iTheta, theta in enumerate(theta_true[:]):
        print('iTheta = ' + str(iTheta+1) + ', true value = ' + str(theta))
        Thetas_ODE = theta_true.copy()
        inner_cost_direction = [[], []]
        outer_cost_direction = [[], []]
        grad_cost_direction = [[], []]
        RMSE_direction = [[], []]
        evaluations_direction = [[], []]
        runtimes_direction = [[], []]
        thetas_checked_direction = [[], []]
        # explore two directions from the truth
        for iDir, direction in enumerate(directions):
            if direction == 'positive':
                range_theta_direction = np.linspace(theta, theta + 3 * sigma, nSamples)
                print('explore positive direction from the truth')
                iPositive  = iDir
            elif direction == 'negative':
                range_theta_direction = np.linspace(theta, theta - 3 * sigma, nSamples)
                print('explore negative direction from the truth')
                # reinitialise at the solution obtained at the truth for the negative direction
                init_betas_roi = init_betas_at_truth.copy()
                iNegative = iDir
            else:
                print('Error with direction settings. Check that you specified all directions you wish to explore correctly')
            if iDir > 0:
                # when we explore negative direction, there is no need to evaluate at truth again
                range_theta_direction_check = np.delete(range_theta_direction, 0)
            ## from here we start the optimisation
            optimisationFailed = False  # crete a flag that is raised when we catch any exception during the optimisation
            ## (might be divergent if we move too far away from the truth
            for iSample, theta_changed in enumerate(range_theta_direction):
                tic_sample = tm.time()
                Thetas_ODE[iTheta] = theta_changed.copy()
                # run the optimisation several times to assess the success rate
                betas_sample = []
                inner_costs_sample = []
                outer_costs_sample = []
                grad_costs_sample = []
                evaluations_sample = []
                runtimes_sample = []
                fitted_state_sample = []
                RMSE_sample = []
                runFailed = [False] * nRuns
                theta_runs = np.array([Thetas_ODE] * nRuns)
                # parallelise the runs for the same sample
                with mp.get_context('fork').Pool(processes=min(ncpu, ncores)) as pool:
                    # package results is a list of tuples
                    results = pool.starmap(inner_optimisation,
                                           zip(theta_runs, repeat(times), repeat(voltage), repeat(output),
                                               repeat(knots), repeat(state_known), repeat(init_betas)))
                # unpack the results
                for iRun, result in enumerate(results):
                    betas_run, inner_cost_run, outer_cost_run, grad_cost_run, evaluations_run, runtime_run, state_fitted_at_run, optimisationFailed = result
                    runFailed[iRun] = optimisationFailed
                    # only add results to the sample if the run was successful
                    if ~optimisationFailed:
                        # get the states at this sample
                        if len(state_fitted_at_run.items()) > 1:
                            list_of_states = [state_values for _, state_values in state_fitted_at_run.items()]
                            state_all_segments = np.array(list_of_states)
                        else:
                            state_all_segments = np.array(state_fitted_at_run[hidden_state_names])
                        # evaluate the cost functions at the sampled value of ODE parameter vector
                        MSE = np.square(np.subtract(state_hidden_true, state_all_segments)).mean()
                        RMSE = np.sqrt(MSE)
                        # store the costs
                        betas_sample.append(betas_run)
                        fitted_state_sample.append(state_all_segments)
                        inner_costs_sample.append(inner_cost_run)
                        outer_costs_sample.append(outer_cost_run)
                        grad_costs_sample.append(grad_cost_run)
                        RMSE_sample.append(RMSE)
                        evaluations_sample.append(evaluations_run)
                        runtimes_sample.append(runtime_run)
                    else:
                        print(str(iRun + 1) + '-th run failed.')
                ## end of loop over runs
                ########################################################################################################
                toc_sample = tm.time()
                if all(runFailed):
                    print('Optimisation at the sampled value = ' + str(
                        theta_changed) + ' failed for all runs. Exploration in this direction is stopped.')
                    break  # sample exploration
                # store all cost values in lists for this particular value of theta
                inner_cost_direction[iDir].append(inner_costs_sample)
                outer_cost_direction[iDir].append(outer_costs_sample)
                grad_cost_direction[iDir].append(grad_costs_sample)
                RMSE_direction[iDir].append(RMSE_sample)
                evaluations_direction[iDir].append(evaluations_sample)
                runtimes_direction[iDir].append(runtimes_sample)
                thetas_checked_direction[iDir].append(theta_changed)
                # print output for the checked value
                print('Value checked: ' + str(theta_changed) + '. Average inner cost across ' + str(
                    nRuns) + ' runs: ' + str(
                    np.nanmean(inner_costs_sample)) + '. Number of evaluations: ' + str(
                    sum(evaluations_sample)) + '. Elapsed time: ' + str(
                    toc_sample - tic_sample) + 's.')
                # find the best Betas out of several runs based on the inner cost]
                index_best_betas = inner_costs_sample.index(np.nanmin(inner_costs_sample))
                # assign values at this index as the initial values for the neighbouring point in the ODE parameter space
                init_betas = betas_sample[index_best_betas]
                # if it is evaluated at the true value of the parameter, save the initial betas to be used in (mu-3sigma,mu) interval
                if iSample == 0 & iDir == 0:
                    init_betas_at_truth = betas_sample[index_best_betas]
            ## end of loop over values of a single theta in unknonw parameters
            ############################################################################################################
            #if the exploration direction was negative, we reverse the list of stored values in case we need to
            # plot anything as a function of monotonously increasing theta
            if direction == 'negative':
                inner_cost_direction[iDir].reverse()
                outer_cost_direction[iDir].reverse()
                grad_cost_direction[iDir].reverse()
                RMSE_direction[iDir].reverse()
                evaluations_direction[iDir].reverse()
                runtimes_direction[iDir].reverse()
                thetas_checked_direction[iDir].reverse()
        #end exploring in both directions
        print('Both directions explored for theta_' + str(iTheta + 1))
        # store all results for plotting
        range_theta = thetas_checked_direction[iNegative] + thetas_checked_direction[iPositive]
        inner_cost_store = inner_cost_direction[iNegative] + inner_cost_direction[iPositive]
        outer_cost_store = outer_cost_direction[iNegative] + outer_cost_direction[iPositive]
        grad_cost_store = grad_cost_direction[iNegative] + grad_cost_direction[iPositive]
        RMSE_store = RMSE_direction[iNegative] + RMSE_direction[iPositive]
        evaluations_store = evaluations_direction[iNegative] + evaluations_direction[iPositive]
        runtimes_store = runtimes_direction[iNegative] + runtimes_direction[iPositive]
        # store best run indeces
        best_run_index = []
        for iPoint in range(len(range_theta)):
            inner_cost_of_joint_segments = inner_cost_store[iPoint]
            best_run_index.append(inner_cost_of_joint_segments.index(min(inner_cost_of_joint_segments)))
        explore_costs[keys[key_counter]] = [range_theta, inner_cost_store, outer_cost_store, grad_cost_store, RMSE_store, evaluations_store, runtimes_store, best_run_index]
        key_counter += 1
    ## end of loop over unknown parameters
    ########################################################################################################################
    print('pause here. see variables')
    # save all results to file
    metadata = {'times': times, 'lambda': lambd, 'state_name': state_name, 'state_true': state_hidden_true,
                'state_known': state_known,
                'knots': knots_roi, 'truth': theta_true, 'param_names': param_names, 'nruns': nRuns, 'param_names': param_names, 'log_scaled': inLogScale}
    with open("Pickles/explore_parameter_whole_" + state_name + ".pkl", "wb") as output_file:
        pkl.dump([explore_costs, metadata], output_file)


