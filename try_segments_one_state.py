# imports
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

def optimise_first_segment(roi,input_roi,output_roi,support_roi,state_known_roi):
    nOutputs = 3
    # define a class that outputs only b-spline surface features
    class bsplineOutputSegment(pints.ForwardModel):
        # this model outputs the discrepancy to be used in a rectangle quadrature scheme
        def simulate(self, parameters, times):
            # given times and return the simulated values
            tck = (support_roi, parameters, degree)
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
    # define a class that outputs only b-spline surface features
    model_bsplines = bsplineOutputSegment()
    values_to_match_output_dims = np.transpose(np.array([output_roi, input_roi, state_known_roi]))
    problem_inner = pints.MultiOutputProblem(model=model_bsplines, times=roi, values=values_to_match_output_dims)
    error_inner = InnerCriterion(problem=problem_inner)
    boundaries_betas = pints.RectangularBoundaries(np.zeros_like(init_betas), 0.99 * np.ones_like(init_betas))
    optimiser_inner = pints.OptimisationController(error_inner, x0=init_betas, sigma0=sigma0_betas,
                                                   boundaries=boundaries_betas, method=pints.CMAES)
    optimiser_inner.set_max_iterations(30000)
    # optimiser_inner.method().set_population_size(min(5, len(init_betas)/2))
    optimiser_inner.set_max_unchanged_iterations(iterations=50, threshold=1e-10)
    optimiser_inner.set_parallel(False)
    optimiser_inner.set_log_to_screen(False)
    betas_roi, cost_roi = optimiser_inner.run()
    nEvaluations = optimiser_inner._evaluations
    return betas_roi, cost_roi, nEvaluations

def optimise_segment(roi,input_roi,output_roi,support_roi,state_known_roi):
    nOutputs = 3
    # define a class that outputs only b-spline surface features
    class bsplineOutputSegment(pints.ForwardModel):
        # this model outputs the discrepancy to be used in a rectangle quadrature scheme
        def simulate(self, parameters, times):
            parameters_with_first = np.insert(parameters, 0, first_spline_coeff)
            # given times and return the simulated values
            tck = (support_roi, parameters_with_first, degree)
            dot_ = sp.interpolate.splev(times, tck, der=1)
            fun_ = sp.interpolate.splev(times, tck, der=0)
            # the RHS must be put into an array
            rhs = ion_channel_model_one_state(times, fun_, Thetas_ODE)
            return np.array([fun_, dot_, rhs]).T

        def n_parameters(self):
            # Return the dimension of the parameter vector
            return nBsplineCoeffs-1

        def n_outputs(self):
            # Return the dimension of the output vector
            return nOutputs
    # define a class that outputs only b-spline surface features
    model_bsplines = bsplineOutputSegment()
    values_to_match_output_dims = np.transpose(np.array([output_roi, input_roi, state_known_roi]))
    problem_inner = pints.MultiOutputProblem(model=model_bsplines, times=roi, values=values_to_match_output_dims)
    error_inner = InnerCriterion(problem=problem_inner)
    boundaries_betas = pints.RectangularBoundaries(np.zeros_like(init_betas), 0.99 * np.ones_like(init_betas))
    optimiser_inner = pints.OptimisationController(error_inner, x0=init_betas, sigma0=sigma0_betas,
                                                   boundaries=boundaries_betas, method=pints.CMAES)
    optimiser_inner.set_max_iterations(30000)
    # optimiser_inner.method().set_population_size(min(5, len(init_betas)/2))
    optimiser_inner.set_max_unchanged_iterations(iterations=50, threshold=1e-10)
    optimiser_inner.set_parallel(False)
    optimiser_inner.set_log_to_screen(False)
    betas_roi, cost_roi = optimiser_inner.run()
    nEvaluations = optimiser_inner._evaluations
    coeffs_with_first = np.insert(betas_roi,0,first_spline_coeff)
    return coeffs_with_first, cost_roi, nEvaluations

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

    # tlim = [0, int(volt_times[-1]*1000)]
    tlim = [3500, 6100]
    times = np.linspace(*tlim, tlim[-1]-tlim[0],endpoint=False)
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
    solution = sp.integrate.solve_ivp(ion_channel_model, [0,tlim[-1]], x0, args=[thetas_true], dense_output=True,method='LSODA',rtol=1e-8,atol=1e-8)
    x_ar = solution.sol(times)
    current_true = observation(times, x_ar, thetas_true)

    # ## single state model
    # # use a as unknown state
    # theta_true = [np.log(2.26e-4), np.log(0.0699), np.log(3.45e-5), np.log(0.05462)]
    # inLogScale = True
    # param_names = ['p_1', 'p_2', 'p_3', 'p_4']
    # a0 = [0]
    # ion_channel_model_one_state = ode_a_only
    # solution_one_state = sp.integrate.solve_ivp(ion_channel_model_one_state, [0,tlim[-1]], a0, args=[theta_true],
    #                                             dense_output=True, method='LSODA',
    #                                             rtol=1e-8, atol=1e-8)
    # state_known_index = state_names.index('r')  # assume that we know r
    # state_known = x_ar[state_known_index, :]
    # state_name = hidden_state_names = 'a'

    ## use r as unknown state
    ## theta_true = [0.0873, 5.15e-3]
    ## inLogScale = False
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
    state_name = hidden_state_names= 'r'

    ################################################################################################################
    ## B-spline representation setup
    # set times of jumps and a B-spline knot sequence
    nPoints_closest = 24  # the number of points from each jump where knots are placed at the finest grid
    nPoints_between_closest = 4  # step between knots at the finest grid
    nPoints_around_jump = 48  # the time period from jump on which we place medium grid
    step_between_knots = 12  # this is the step between knots around the jump in the medium grid
    nPoints_between_jumps = 2  # this is the number of knots at the coarse grid corresponding to slowly changing values
    ## find switchpoints
    d2v_dt2 = np.diff(volts_new, n=2)
    dv_dt = np.diff(volts_new)
    der1_nonzero = np.abs(dv_dt) > 1e-6
    der2_nonzero = np.abs(d2v_dt2) > 1e-6
    switchpoints = [a and b for a, b in zip(der1_nonzero, der2_nonzero)]
    # ignore everything outside of the region of iterest
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
    ## create multiple segments limited by time instances of jumps
    times_roi = []
    states_roi = []
    states_known_roi = []
    current_roi = []
    voltage_roi = []
    knots_roi = []
    collocation_roi = []
    colderiv_roi = []
    for iJump, jump in enumerate(jump_indeces[:-1]): #loop oversegments (nJumps - )
        # define a region of interest - we will need this to preserve the
        # trajectories of states given the full clamp and initial position, while
        ROI_start = jump
        ROI_end = jump_indeces[iJump+1]+1 # add one to ensure that t_end equals to t_start of the following segment
        ROI = times[ROI_start:ROI_end]
        x_ar = solution.sol(ROI)
        # get time points to compute the fit to ODE cost
        times_roi.append(ROI)
        # save states
        states_roi.append(solution_one_state.sol(ROI))
        states_known_roi.append(x_ar[state_known_index,:])
        # save current
        current_roi.append(observation(ROI, x_ar, thetas_true))
        # save voltage
        voltage_roi.append(V(ROI))
        ## add colloation points
        abs_distance_lists = [[(num - index) for num in range(ROI_start,ROI_end)] for index in
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
        knots_jump = [a or b for a, b in zip(knots_after_jump, knots_before_jump)] # logical sum of mininal and maximal distances
        # convert to numeric array again
        knot_indeces = [i + ROI_start for i, x in enumerate(knots_jump) if x]
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
        indeces_inner.sort()  # sort list in ascending order - this is done inplace
        degree = 3
        # define the Boor points to
        indeces_outer = [indeces_inner[0]]*3 + [indeces_inner[-1]]*3
        boor_indeces = np.insert(indeces_outer, degree, indeces_inner)  # create knots for which we want to build splines
        knots = times[boor_indeces]
        # save knots for the segment - including additional points at the edges
        knots_roi.append(knots)
        # build the collocation matrix using the defined knot structure
        coeffs = np.zeros(len(knots)-degree-1)  # number of splines will depend on the knot order
        spl_ones = BSpline(knots, np.ones_like(coeffs), degree)
        # spl_deriv = spl_ones.derivative(nu=1) # this is a definition of derivative - might come handy later for cost evaluation on the whole time-series
        # spl_deriv.c[:] = 1. # change coefficients to ones to make sure we can get collocation differentiation matrix
        # tau = np.arange(knots[0], knots[-1])
        splinest = [None] * len(coeffs)
        splineder = [None] * len(coeffs)# the grid of indtividual splines is required to generate a collocation matrix
        for i in range(len(coeffs)):
            coeffs[i] = 1.
            splinest[i] = BSpline(knots, coeffs.copy(), degree, extrapolate=False)  # create a spline that only has one non-zero coeff
            coeffs[i] = 0.
        collocation_roi.append(collocm(splinest, ROI))
    ##^ this loop stores the time intervals from which to draw collocation points and the data for piece-wise fitting
    ####################################################################################################################
    ## define pints classes for optimisation
    ## Classes to run optimisation in pints
    lambd = 1 #0.000001
    ## Classes to run optimisation in pints
    nBsplineCoeffs = len(coeffs)  # this to be used in params method of class ForwardModel
    init_betas = 0.5 * np.ones(nBsplineCoeffs)  # initial values of B-spline coefficients
    sigma0_betas = 0.2 * np.ones(nBsplineCoeffs) # sigma of inital spread of values
    Thetas_ODE = theta_true.copy()
    print('Number of B-spline coeffs: ' + str(nBsplineCoeffs))
    nOutputs = 3
    # define a class that outputs only b-spline surface features
    class bsplineOutputTest(pints.ForwardModel):
        # this model outputs the discrepancy to be used in a rectangle quadrature scheme
        def simulate(self, parameters, support, times):
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
            model_output = self._problem.evaluate(
                betas)  # the output of the model with be an array of size nTimes x nOutputs
            x, x_dot, rhs = np.split(model_output, 3, axis=1)  # we split the array into states, state derivs, and RHSs
            # compute the data fit
            volts_for_model = self._values[:, 1]  # we need to make sure that voltage is read at the times within ROI so we pass it in as part of values
            d_y = g * x[:, 0] * self._values[:, 2] * (volts_for_model - EK) - self._values[:, 0]
            data_fit_cost = np.transpose(d_y) @ d_y
            # compute the gradient matching cost
            d_deriv = (x_dot[:] - rhs[:]) ** 2
            integral_quad = sp.integrate.simpson(y=d_deriv, even='avg', axis=0)
            gradient_match_cost = np.sum(integral_quad, axis=0)
            # not the most elegant implementation because it just grabs global lambda
            return data_fit_cost + lambd * gradient_match_cost
    ####################################################################################################################
    # try optimising several segments
    nRuns = 1
    cost_threshold = 10**(-3)
    # store the true state
    times_of_segments = np.hstack(times_roi[:len(times_roi)])
    state_hidden_true = solution_one_state.sol(times_of_segments)[0, :]
    states_of_segments_known = np.hstack(states_known_roi[:len(times_roi) + 1])
    # initialise the figure with true values
    fig, axes = plt.subplot_mosaic([['a)'], ['b)']], layout='constrained')
    y_labels = ['I', '$' + state_name + '$']
    axes['a)'].plot(times_of_segments, observation(times_of_segments, solution.sol(times_of_segments), thetas_true), '-k', label=r'Current true',
                    linewidth=2, alpha=0.7)
    axes['b)'].plot(times_of_segments, state_hidden_true, '-k', label=r'$' + state_name + '$ true', linewidth=2,
                    alpha=0.7)
    # run the optimisation 20 times to assess the success rate
    betas_all_runs = []

    costs_all_runs = []
    evaluations_all_runs = []
    fitted_state_all_runs = []
    RMSE_all_runs = []
    sucsess_rate_all_runs = []
    for iRun in range(nRuns):
        all_betas = []
        all_costs = []
        *ps, g = thetas_true
        model_bsplines = bsplineOutputTest()
        end_of_roi = []
        state_fitted_roi = {key: [] for key in state_names}
        totalEvaluations = []
        big_tic = tm.time()
        for iSegment in range(1):
            tic = tm.time()
            segment = times_roi[iSegment]
            input_segment = voltage_roi[iSegment]
            output_segment = current_roi[iSegment]
            support_segment = knots_roi[iSegment]
            state_known_segment = states_known_roi[iSegment]
            init_betas = 0.5 * np.ones(nBsplineCoeffs)
            sigma0_betas = 0.2 * np.ones(nBsplineCoeffs)
            betas, cost, nEvals = optimise_first_segment(segment,input_segment,output_segment,support_segment,state_known_segment)
            totalEvaluations.append(nEvals)
            all_betas.append(betas)
            all_costs.append(cost)
            toc = tm.time()
            print('Segment ' + str(iSegment+1) + ' is finished after '+ str(nEvals) +' evaluations. Elapsed time: ' + str(toc-tic) + 's.')
            # check collocation solution against truth
            model_output_fit_at_truth = model_bsplines.simulate(betas,support_segment, segment)
            state_at_truth, state_deriv_at_truth, rhs_truth = np.split(model_output_fit_at_truth, 3, axis=1)
            current_model_at_truth = g * state_at_truth[:, 0] * state_known_segment * (input_segment - EK)
            # ## uncomment the line below to plot the cost terms of a given segment
            # plot_segment(iSegment, segment, output_segment, current_model_at_truth, model_output_fit_at_truth,
            #              states_roi[iSegment][0, :], state_name)
            # save the final value of the segment
            end_of_roi.append(state_at_truth[-1,:])
            for iState, stateName in enumerate(hidden_state_names):
                state_fitted_roi[stateName] += list(state_at_truth[:, iState])
        ####################################################################################################################
        #  optimise the following segments by matching the first B-spline height to the previous segment
        for iSegment in range(1,len(times_roi)):
            tic = tm.time()
            segment = times_roi[iSegment]
            input_segment = voltage_roi[iSegment]
            output_segment = current_roi[iSegment]
            support_segment = knots_roi[iSegment]
            collocation_segment = collocation_roi[iSegment]
            state_known_segment = states_known_roi[iSegment]
            # find the scaling coeff of the first height by matiching its height at t0 of the segment to the final value of the previous segment
            first_spline_coeff = end_of_roi[-1][0] / collocation_segment[0, 0]
            # we must re-initalised the optimisation with that excludes the first coefficient
            init_betas = 0.5 * np.ones(nBsplineCoeffs - 1)  # initial values of B-spline coefficients
            sigma0_betas = 0.2 * np.ones(nBsplineCoeffs - 1) # inital spread of values
            betas, cost, nEvals = optimise_segment(segment,input_segment,output_segment,support_segment,state_known_segment)
            totalEvaluations.append(nEvals)
            all_betas.append(betas)
            all_costs.append(cost)
            toc = tm.time()
            print('Segment ' + str(iSegment) + ' is finished after '+ str(nEvals) +' evaluations. Elapsed time: ' + str(toc-tic) + 's.')
            # check collocation solution against truth
            model_output_fit_at_truth = model_bsplines.simulate(betas, support_segment, segment)
            state_at_truth, state_deriv_at_truth, rhs_truth = np.split(model_output_fit_at_truth, 3, axis=1)
            current_model_at_truth = g * state_at_truth[:, 0] * state_known_segment * (input_segment - EK)
            # ## uncomment the line below to plot the cost terms of a given segment
            # plot_segment(iSegment, segment, output_segment, current_model_at_truth, model_output_fit_at_truth, states_roi[iSegment][0, :], state_name)
            # store end of segment and the whole state for the
            end_of_roi.append(state_at_truth[-1, :])
            for iState, stateName in enumerate(hidden_state_names):
                state_fitted_roi[stateName] += list(state_at_truth[:, iState])
        #### end of loop over segments
        ####################################################################################################################
        big_toc = tm.time()
        print('Total evaluations: ' + str(sum(totalEvaluations)) + '. Total runtime: ' + str(big_toc-big_tic) + ' s.' )
        states_of_segments_hidden = state_fitted_roi[state_name]
        # compute the prediction error
        MSE = np.square(np.subtract(state_hidden_true, states_of_segments_hidden)).mean()
        RMSE = np.sqrt(MSE)
        # see how many segments were fitted with the cost lower than threshold
        success_rate =  sum([1 for i, cost in enumerate(all_costs) if cost < cost_threshold])/len(all_costs)
        # store results of the run
        betas_all_runs.append(all_betas)
        costs_all_runs.append(all_costs)
        evaluations_all_runs.append(totalEvaluations)
        fitted_state_all_runs.append(states_of_segments_hidden)
        RMSE_all_runs.append(RMSE)
        sucsess_rate_all_runs.append(success_rate)
    ####################################################################################################################
        cost_all_segments = [sum(list) for list in costs_all_runs]
    # plot joint segments
        axes['a)'].plot(times_of_segments,
                        observation(times_of_segments, np.array([states_of_segments_known, states_of_segments_hidden]),
                                    thetas_true), '--c',alpha=0.7)
        axes['b)'].plot(times_of_segments, state_fitted_roi[state_name], '--c',alpha=0.7)
        if iSegment == len(times_roi)-1:
            axes['a)'].plot(times_of_segments, observation(times_of_segments, np.array([states_of_segments_known,states_of_segments_hidden]), thetas_true), '--b', label=r'Optimised given true $\theta$')
            axes['b)'].plot(times_of_segments, state_fitted_roi[state_name], '--c', label=r'B-spline approximation given true $\theta$')
    ### end loop
    ### display legend in the figure comparing true state to its estimates
    iAx = 0
    # only assign ledgend to the last run
    display = (0,nRuns)
    for _, ax in axes.items():
        ax.set_ylabel(y_labels[iAx], fontsize=12)
        handles, labels = ax.get_legend_handles_labels()
        ax.legend([handle for i, handle in enumerate(handles) if i in display],[label for i, label in enumerate(labels) if i in display], fontsize=12, loc='upper left')
        iAx += 1
    plt.tight_layout(pad=0.3)
    plt.savefig('Figures/all_segments_state_'+ state_name +'_nruns_'+str(nRuns)+'.png', dpi=400)


    # plot performace of algorithm
    fig, axes = plt.subplots(2,1,figsize=(10,10),sharex=True)
    axes[0].plot(RMSE_all_runs,marker='o',linestyle = 'None')
    axes[0].set_yscale('log')
    axes[0].set_ylabel('RMSE of state estimate')
    axes[1].plot(sucsess_rate_all_runs,marker='o',linestyle = 'None')
    axes[1].set_ylabel('n of optimisations reaching <'+str(cost_threshold)+'')
    axes[1].set_xlabel('run index')
    plt.tight_layout(pad=0.3)
    plt.savefig('Figures/performance_'+ state_name +'.png', dpi=400)


    fig,axes = plt.subplots(2,1,figsize=(10,10),sharex=True)
    axes[1].plot(cost_threshold * np.ones(iSegment+1),'-k',lw=1)
    for iRun in range(nRuns-1):
        axes[0].plot(evaluations_all_runs[iRun], marker='o', linestyle='None')
        axes[1].plot(costs_all_runs[iRun],marker='o',linestyle = 'None')
    axes[0].set_ylabel('n of evaluations')
    axes[1].set_ylabel('cost at segment')
    axes[1].set_yscale('log')
    axes[1].set_xlabel('segment index')
    plt.tight_layout(pad=0.3)
    plt.savefig('Figures/evaluations_and_costs_' + state_name + '.png', dpi=400)


    # store the metadata of this optimisation
    metadata = {'times': times, 'lambda': lambd, 'state_name': state_name, 'state_true': state_hidden_true,
                'state_known': states_of_segments_known,
                'knots': knots_roi, 'truth': theta_true, 'param_names': param_names, 'log_scaled': inLogScale}
    # store results
    opt_results = {'betas': betas_all_runs, 'costs': costs_all_runs, 'evaluations':evaluations_all_runs,'success_rates':sucsess_rate_all_runs,'arbitrary cost threshold': cost_threshold}
    with open("Pickles/optimisation_of_segments_performance_" + state_name + ".pkl", "wb") as output_file:
        pkl.dump([opt_results, metadata], output_file)
    print('pause here')
