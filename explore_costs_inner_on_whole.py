# imports
import numpy as np

from setup import *
import pints
import pickle as pkl
matplotlib.use('AGG')
plt.ioff()

# definitions
def ion_channel_model(t, x, theta):
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

# observation model for two gating variables
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
    tlim = [3500, 6100]
    times = np.linspace(*tlim, tlim[-1])
    volts_new = V(times)
    ## Generate the synthetic data
    # parameter values for the model
    EK = -80
    thetas_true = [2.26e-4, 0.0699, 3.45e-5, 0.05462, 0.0873, 8.91e-3, 5.15e-3, 0.03158, 0.1524]
    # initialise and solve ODE
    x0 = [0, 1]
    state_names = ['a','r']
    # solve initial value problem
    tlim[1]+=1 #solve for a slightly longer period
    solution = sp.integrate.solve_ivp(ion_channel_model, tlim, x0, args=[thetas_true], dense_output=True,method='LSODA',rtol=1e-8,atol=1e-8)
    x_ar = solution.sol(times)
    output = observation(times, x_ar, thetas_true)

    ## single state models
    ######################
    # use a as unknown state
    state_name = hidden_state_names= 'a'
    theta_true = [np.log(2.26e-4), np.log(0.0699), np.log(3.45e-5), np.log(0.05462)]
    inLogScale = True
    param_names = ['p_1','p_2','p_3','p_4']
    a0 = [0]
    ion_channel_model_one_state = ode_a_only
    solution_one_state = sp.integrate.solve_ivp(ion_channel_model_one_state, tlim, a0, args=[theta_true], dense_output=True, method='LSODA',
                                      rtol=1e-8, atol=1e-8)
    state_known_index = state_names.index('r')  # assume that we know r
    state_known = x_ar[state_known_index, :]
    state_hidden_true = x_ar[state_names.index(state_name),:]


    # ## use r as unknown state
    # ## theta_true = [0.0873, 5.15e-3]
    # ## inLogScale = False
    # state_name = hidden_state_names = 'r'
    # theta_true = [np.log(0.0873), np.log(8.91e-3), np.log(5.15e-3), np.log(0.03158)]
    # inLogScale = True
    # param_names = ['p_5','p_6','p_7','p_8']
    # r0 = [1]
    # ion_channel_model_one_state = ode_r_only
    # solution_one_state = sp.integrate.solve_ivp(ion_channel_model_one_state, tlim, r0, args=[theta_true], dense_output=True,
    #                                     method='LSODA',
    #                                     rtol=1e-8, atol=1e-10)
    # state_known_index = state_names.index('a')  # assume that we know a
    # state_known = x_ar[state_known_index,:]
    # state_hidden_true = x_ar[state_names.index(state_name),:]


    ## boundaries of thetas from Clerx et.al. paper - they are the same for two gating variables
    theta_lower_boundary = [np.log(10 ** (-7)), np.log(10 ** (-7)), np.log(10 ** (-7)), np.log(10 ** (-7))]
    theta_upper_boundary = [np.log(10 ** (3)), np.log(0.4), np.log(10 ** (3)), np.log(0.4)]
    ################################################################################################################
    ## find switchpoints
    d2v_dt2 = np.diff(volts_new, n=2)
    dv_dt = np.diff(volts_new)
    der1_nonzero = np.abs(dv_dt) > 1e-6
    der2_nonzero = np.abs(d2v_dt2) > 1e-6
    switchpoints = [a and b for a, b in zip(der1_nonzero, der2_nonzero)]
    # ignore everything outside of the region of iterest
    ROI_start = 3500
    ROI_end = tlim[-1]
    ROI = range(ROI_start, ROI_end)
    ####################################################################################################################
    ## B-spline representation setup
    # set times of jumps and a B-spline knot sequence
    nPoints_closest = 24  # the number of points from each jump where knots are placed at the finest grid
    nPoints_between_closest = 12  # step between knots at the finest grid
    nPoints_around_jump = 48  # the time period from jump on which we place medium grid
    step_between_knots = 48  # this is the step between knots around the jump in the medium grid
    nPoints_between_jumps = 2  # this is the number of knots at the coarse grid corresponding to slowly changing values

    # get the times of all jumps
    a = [0] + [i + 1 for i, x in enumerate(switchpoints) if x] + [
        len(ROI)]  # get indeces of all the switchpoints, add t0 and tend
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
    knots_after_jump = [
        ((x <= 2) and (x % 1 == 0)) or ((x <= nPoints_closest) and (x % nPoints_between_closest == 0)) or (
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
    knots_all = knot_times.copy()  # + close_knots_duplicate_times.copy() # to see if having two splines tied to the close knots will improve precision
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
    outer = [knots_all[0], knots_all[0], knots_all[0], knots_all[-1], knots_all[-1], knots_all[-1]]
    support = np.insert(outer, 3, knots_all)
    nBsplineCoeffs = len(support) - degree - 1 # this to be used in params method of class ForwardModel
    ## ^ this code creates a grid of B-spline knots on the entire region of interest
    ####################################################################################################################
    ## define pints classes for optimisation
    lambd = 1
    ## Classes to run optimisation in pints
    print('Number of B-spline coeffs: ' + str(nBsplineCoeffs))
    nOutputs = 3
    # define a class that outputs only b-spline surface features - we need it to compute outer cost
    class bsplineOutput(pints.ForwardModel):
        # this model outputs the discrepancy to be used in a rectangle quadrature scheme
        def simulate(self, parameters, times):
            # given times and return the simulated values
            tck = (support, parameters, degree)
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
    # try optimising several segments
    nRuns = 9
    cost_threshold = 10**(-3)
    ###############################################################################################################
    ## set up the optimisation
    sigma = 1 # std from thhe true value
    nSamples = 15 # number of samples on either side of the true values
    keys = ['theta_{' + str(index) + '}' for index in np.arange(len(theta_true))]
    explore_costs = dict.fromkeys(keys)
    explore_costs_segment_info = dict.fromkeys(keys)
    key_counter = 0
    model_bsplines = bsplineOutput()
    init_betas = 0.5 * np.ones(nBsplineCoeffs)  # initial values of B-spline coefficients
    sigma0_betas = 0.2 * np.ones(nBsplineCoeffs)
    ## create the problem of comparing the modelled current with measured current
    voltage = V(times)  # must read voltage at the correct times to match the output
    values_to_match_output_dims = np.transpose(np.array([output, voltage, state_known]))
    # ^ we actually only need first two columns in this array but pints wants to have the same number of values and outputs
    problem_inner = pints.MultiOutputProblem(model=model_bsplines, times=times, values=values_to_match_output_dims)
    ## associate the cost with it
    error_inner = InnerCriterion(problem=problem_inner)
    ##  define boundaries for the inner optimisation
    boundaries_betas = pints.RectangularBoundaries(np.zeros_like(init_betas), np.ones_like(init_betas))
    # check all parameter values within range (mu, mu+3sigma)
    for iTheta, theta in enumerate(theta_true[:]):
        print('iTheta = ' + str(iTheta) + ', true value = ' + str(theta))
        Thetas_ODE = theta_true.copy()
        range_theta_plus = np.linspace(theta, theta + 3 * sigma, nSamples)
        inner_cost_plus = []
        outer_cost_plus = []
        grad_cost_plus = []
        RMSE_plus = []
        evaluations_plus = []
        runtimes_plus = []
        for iSample, theta_changed in enumerate(range_theta_plus):
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
            for iRun in range(nRuns):
                *ps, g = thetas_true
                tic_run = tm.time()
                ####################################################################################################################
                #  optimise against the whole time-serise
                optimiser_inner = pints.OptimisationController(error_inner, x0=init_betas, sigma0=sigma0_betas,
                                                               boundaries=boundaries_betas, method=pints.CMAES)
                optimiser_inner.set_max_iterations(30000)
                optimiser_inner.set_max_unchanged_iterations(iterations=50, threshold=1e-8)
                optimiser_inner.set_parallel(False)
                optimiser_inner.set_log_to_screen(False)
                Betas_BSPL, inner_cost = optimiser_inner.run()
                evals = optimiser_inner._evaluations
                #re-initialise betas for the next run
                init_betas = Betas_BSPL.copy()
                sigma0_betas = 0.2 * np.ones(nBsplineCoeffs)  # inital spread of values
                # check collocation solution against truth
                model_output = model_bsplines.simulate(Betas_BSPL, times)
                state_at_estimate, deriv_at_estimate, rhs_at_estimate = np.split(model_output, 3, axis=1)
                current_model_at_estimatete = g * state_at_estimate[:, 0] * state_known * (voltage - EK)
                dy = (current_model_at_estimatete - output)
                d_deriv = np.square(np.subtract(deriv_at_estimate, rhs_at_estimate))
                integral_quad = sp.integrate.simpson(y=d_deriv, even='avg', axis=0)
                # compute outer cost and gradient matching cost
                gradient_cost = np.sum(integral_quad, axis=0)
                outer_cost = dy @ np.transpose(dy)
                # add all costs and performance metrics to store for the run
                #### end of loop over segments
                ####################################################################################################################
                toc_run = tm.time()
                print(str(iRun+1) +'-th run complete. Total evaluations: ' + str(evals) + '. Total runtime: ' + str(toc_run-tic_run) + ' s.' )
                # compute the prediction error
                MSE = np.square(np.subtract(state_hidden_true, state_at_estimate)).mean()
                RMSE = np.sqrt(MSE)
                # see how many segments were fitted with the cost lower than threshold
                # store results of the run
                betas_sample.append(Betas_BSPL)
                inner_costs_sample.append(inner_cost)
                outer_costs_sample.append(outer_cost)
                grad_costs_sample.append(gradient_cost)
                fitted_state_sample.append(state_at_estimate)
                RMSE_sample.append(RMSE)
                evaluations_sample.append(evals)
                runtimes_sample.append(toc_run - tic_run)
                ### end loop over segments
            ## end of loop over runs
            toc_sample = tm.time()
            # store all cost values in lists for this particular value of theta
            inner_cost_plus.append(inner_costs_sample)
            outer_cost_plus.append(outer_costs_sample)
            grad_cost_plus.append(grad_costs_sample)
            RMSE_plus.append(RMSE_sample)
            evaluations_plus.append(evaluations_sample)
            # print output for the checked value
            print('Value checked: ' + str(theta_changed) + '. Average inner cost across ' + str(nRuns) + ' runs: ' + str(
                np.mean(inner_costs_sample)) + '. Number of evaluations: ' + str(sum(evaluations_sample)) + '. Elapsed time: ' + str(
                toc_sample - tic_sample) + 's.')
            # find the best Betas out of several runs based on the inner cost]
            index_best_betas = inner_costs_sample.index(min(inner_costs_sample))
            # assign values at this index as the initial values for the neighbouring point in the ODE parameter space
            init_betas_roi = betas_sample[index_best_betas]
            # if it is evaluated at the true value of the parameter, save the initial betas to be used in (mu-3sigma,mu) interval
            if iSample == 0:
                init_betas_at_truth = betas_sample[index_best_betas]
        ## end of loop over values of a single theta in unknonw parameters
        print('Change of direction w.r.t. truth')
        ## now check the values in the opposite direction from the truth
        range_theta_minus = np.linspace(theta, theta - 3 * sigma, nSamples)
        inner_cost_minus = []
        outer_cost_minus = []
        grad_cost_minus = []
        RMSE_minus = []
        evaluations_minus = []
        # reinitialise at the solution obtained at the truth
        init_betas_roi = init_betas_at_truth.copy()
        # for iSample, theta_changed in enumerate(range_theta_minus[1:]): # skip the minima as we have already evaluated it
        #     tic_sample = tm.time()
        #     Thetas_ODE[iTheta] = theta_changed.copy()
        #     # run the optimisation several times to assess the success rate
        #     betas_sample = []
        #     inner_costs_sample = []
        #     outer_costs_sample = []
        #     grad_costs_sample = []
        #     evaluations_sample = []
        #     fitted_state_sample = []
        #     RMSE_sample = []
        #     sucsess_rate_sample = []
        #     #########################
        #     ## placeholder forr the optimisation here
        #     #########################
        #     toc_sample = tm.time()
        #     total_inner_cost_sample = [sum(list) for list in inner_costs_sample]
        #     total_outer_cost_sample = [sum(list) for list in outer_costs_sample]
        #     total_grad_cost_sample = [sum(list) for list in grad_costs_sample]
        #     total_evals_sample = [sum(list) for list in evaluations_sample]
        #     # store all cost values in lists for this particular value of theta
        #     inner_cost_minus.append(total_inner_cost_sample)
        #     outer_cost_minus.append(total_outer_cost_sample)
        #     grad_cost_minus.append(total_grad_cost_sample)
        #     RMSE_minus.append(RMSE_sample)
        #     evaluations_minus.append(total_evals_sample)
        #     # print output for the checked value
        #     print(
        #         'Value checked: ' + str(theta_changed) + '. Average inner cost across ' + str(nRuns) + ' runs: ' + str(
        #             np.mean(total_inner_cost_sample)) + '. Number of evaluations: ' + str(sum(
        #             total_evals_sample)) + '. Elapsed time: ' + str(
        #             toc_sample - tic_sample) + 's.')
        #     # find the best Betas out of several runs based on the inner cost]
        #     index_best_betas = total_inner_cost_sample.index(min(total_inner_cost_sample))
        #     # assign values at this index as the initial values for the neighbouring point in the ODE parameter space
        #     init_betas_roi = betas_sample[index_best_betas]
        ## end of loop over parameter values
        print('Both directions explored for theta_'+str(iTheta+1))
        # combine lists to get the interval (mu-3sigma, mu+3sigma)
        inner_cost_minus.reverse()
        outer_cost_minus.reverse()
        grad_cost_minus.reverse()
        RMSE_minus.reverse()
        evaluations_minus.reverse()
        range_theta = list(np.flip(range_theta_minus[1:])) + list(range_theta_plus)
        inner_cost_store = inner_cost_minus + inner_cost_plus
        outer_cost_store = outer_cost_minus + outer_cost_plus
        grad_cost_store = grad_cost_minus + grad_cost_plus
        RMSE_store = RMSE_minus + RMSE_plus
        evaluations_store = evaluations_minus + evaluations_plus

        # store best run indeces
        best_run_index = []
        for iPoint in range(len(range_theta)):
            inner_cost_of_joint_segments = inner_cost_store[iPoint]
            best_run_index.append(inner_cost_of_joint_segments.index(min(inner_cost_of_joint_segments)))
        explore_costs[keys[key_counter]] = [range_theta, inner_cost_store, outer_cost_store, grad_cost_store, RMSE_store, evaluations_store, best_run_index]
        key_counter += 1
    ## end of loop over unknown parameters
    ########################################################################################################################
    print('pause here. see variables')
    # save all results to file
    metadata = {'times': times, 'lambda': lambd, 'state_name': state_name, 'state_true': state_hidden_true,
                'state_known': state_known,
                'knots': knots, 'truth': theta_true, 'param_names': param_names, 'nruns': nRuns, 'param_names': param_names, 'log_scaled': inLogScale}
    with open("Pickles/explore_parameter_space_whole_" + state_name + ".pkl", "wb") as output_file:
        pkl.dump([explore_costs, metadata], output_file)


