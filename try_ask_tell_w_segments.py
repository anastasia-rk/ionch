# imports
import numpy as np

from setup import *
import pints
import pickle as pkl
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

class BoundariesOneState(pints.Boundaries):
    """
    Boundary constraints on the parameters for a single state variable

    """

    def __init__(self):

        super(BoundariesOneState, self).__init__()

        # Limits on p1-p4 for a signle gative variable model
        self.lower_alpha = 1e-7  # Kylie: 1e-7
        self.upper_alpha = 1e3  # Kylie: 1e3
        self.lower_beta = 1e-7  # Kylie: 1e-7
        self.upper_beta = 0.4  # Kylie: 0.4

        # Lower and upper bounds for all parameters
        self.lower = [
            self.lower_alpha,
            self.lower_beta,
            self.lower_alpha,
            self.lower_beta,
        ]
        self.upper = [
            self.upper_alpha,
            self.upper_beta,
            self.upper_alpha,
            self.upper_beta,
        ]

        self.lower = np.array(self.lower)
        self.upper = np.array(self.upper)

        # Limits on maximum reaction rates
        self.rmin = 1.67e-5
        self.rmax = 1000

        # Voltages used to calculate maximum rates
        self.vmin = -120
        self.vmax = 60

    def n_parameters(self):
        return 4

    def check(self, transformed_parameters):

        debug = False

        # # check if parameters are sampled in log space
        # if InLogScale:
        #     # Transform parameters back to decimal space
        #     parameters = np.exp(transformed_parameters)
        # else:
        #     # leave as is
        #     parameters = transformed_parameters

        # Transform parameters back to decimal space
        parameters = np.exp(transformed_parameters)

        # Check parameter boundaries
        if np.any(parameters < self.lower):
            if debug:
                print('Lower')
            return False
        if np.any(parameters > self.upper):
            if debug:
                print('Upper')
            return False

        # Check maximum rate constants
        p1, p2, p3, p4 = parameters[:]

        # Check positive signed rates
        r = p1 * np.exp(p2 * self.vmax)
        if r < self.rmin or r > self.rmax:
            if debug:
                print('r1')
            return False

        # Check negative signed rates
        r = p3 * np.exp(-p4 * self.vmin)
        if r < self.rmin or r > self.rmax:
            if debug:
                print('r3')
            return False

        return True

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
    tlim = [4500, 12000]
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
    state_name = hidden_state_names= 'r'
    ################################################################################################################
    ## store true hidden state
    state_hidden_true = x_ar[state_names.index(state_name), :]
    ## rectangular boundaries of thetas from Clerx et.al. paper - they are the same for two gating variables
    theta_lower_boundary = [np.log(10 ** (-5)), np.log(10 ** (-5)), np.log(10 ** (-5)), np.log(10 ** (-5))]
    theta_upper_boundary = [np.log(10 ** (3)), np.log(0.4), np.log(10 ** (3)), np.log(0.4)]
    ################################################################################################################
    ## B-spline representation setup
    # set times of jumps and a B-spline knot sequence
    nPoints_closest = 24  # the number of points from each jump where knots are placed at the finest grid
    nPoints_between_closest = 8  # step between knots at the finest grid
    nPoints_around_jump = 48  # the time period from jump on which we place medium grid
    step_between_knots = 48  # this is the step between knots around the jump in the medium grid
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
    ## create multiple segments limited by time instances of jumps
    times_roi = []
    states_roi = []
    states_known_roi = []
    current_roi = []
    voltage_roi = []
    knots_roi = []
    collocation_roi = []
    colderiv_roi = []
    init_betas_roi = []
    for iJump, jump in enumerate(jump_indeces[:-1]):  # loop oversegments (nJumps - )
        # define a region of interest - we will need this to preserve the
        # trajectories of states given the full clamp and initial position, while
        ROI_start = jump
        ROI_end = jump_indeces[iJump + 1] + 1  # add one to ensure that t_end equals to t_start of the following segment
        ROI = times[ROI_start:ROI_end]
        x_ar = solution.sol(ROI)
        # get time points to compute the fit to ODE cost
        times_roi.append(ROI)
        # save states
        states_roi.append(solution_one_state.sol(ROI))
        states_known_roi.append(x_ar[state_known_index, :])
        # save current
        current_roi.append(observation(ROI, x_ar, thetas_true))
        # save voltage
        voltage_roi.append(V(ROI))
        ## add colloation points
        abs_distance_lists = [[(num - index) for num in range(ROI_start, ROI_end)] for index in
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
        degree = 3
        # define the Boor points to
        indeces_outer = [indeces_inner[0]] * 3 + [indeces_inner[-1]] * 3
        boor_indeces = np.insert(indeces_outer, degree,
                                 indeces_inner)  # create knots for which we want to build splines
        knots = times[boor_indeces]
        # save knots for the segment - including additional points at the edges
        knots_roi.append(knots)
        # build the collocation matrix using the defined knot structure
        coeffs = np.zeros(len(knots) - degree - 1)  # number of splines will depend on the knot order
        spl_ones = BSpline(knots, np.ones_like(coeffs), degree)
        splinest = [None] * len(coeffs)
        splineder = [None] * len(coeffs)  # the grid of indtividual splines is required to generate a collocation matrix
        for i in range(len(coeffs)):
            coeffs[i] = 1.
            splinest[i] = BSpline(knots, coeffs.copy(), degree,
                                  extrapolate=False)  # create a spline that only has one non-zero coeff
            coeffs[i] = 0.
        collocation_roi.append(collocm(splinest, ROI))
        # create inital values of beta to be used at the true value of parameters
        init_betas_roi.append(0.5 * np.ones_like(coeffs))
    ##^ this loop stores the time intervals from which to draw collocation points and the data for piece-wise fitting # this to be used in params method of class ForwardModel
    nBsplineCoeffs = len(coeffs)
    print('Number of B-spline coeffs per segment: ' + str(nBsplineCoeffs))
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

    # # define a class that outputs only b-spline surfaces and its derivative for a segment
    nThetas = len(theta_true)
    class SegmentOutput(pints.ForwardModel):
        # this model outputs the discrepancy to be used in a rectangle quadrature scheme
        def simulate(self, parameters, times):
            # given segments return the values for a segment
            coeffs = betas_segment
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

    # define an error w.r.t. the ODE parameters that assumes that it knows B-spline parameters - simply data fit
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
            # model_output = self._problem.evaluate(thetas)   # the output of the model with be an array of size nTimes x nOutputs
            # x, x_dot = np.split(model_output, 2, axis=1)
            x = state_all_segments
            # compute the data fit
            d_y = g * x[:] * state_known * (self._values[:,1] - EK) - self._values[:,0] # this part depends on theta_g
            data_fit_cost = np.transpose(d_y) @ d_y
            return data_fit_cost
    ####################################################################################################################
    ## Create objects for the optimisation
    lambd = 1 # 0.3 # 0 # 1
    # set initial values and boundaries depending on the scale of search space
    if inLogScale:
        # theta in log scale
        init_thetas = -5 * np.ones(nThetas)
        sigma0_thetas = 2 * np.ones(nThetas)
        boundaries_thetas = pints.RectangularBoundaries(theta_lower_boundary, theta_upper_boundary)
        boundaries_thetas_Michael = BoundariesOneState()
    else:
        # theta in decimal scale
        init_thetas = 0.001 * np.ones(nThetas)
        sigma0_thetas = 0.0005 * np.ones(nThetas)
        boundaries_thetas = pints.RectangularBoundaries(np.exp(theta_lower_boundary), np.exp(theta_upper_boundary))
    # outer optimisation settings
    ### BEAR IN MIND THAT OUTER OPTIMISATION is conducted on the entire time-series
    model_bsplines_test = bsplineOutput()
    model_segments = SegmentOutput()
    ## create the problem of comparing the modelled current with measured current
    voltage = V(times)  # must read voltage at the correct times to match the output
    current_true = observation(times, solution.sol(times), thetas_true)
    values_to_match_output_ode = np.transpose(np.array([current_true, voltage]))
    # ^ we actually only need first two columns in this array but pints wants to have the same number of values and outputs
    problem_outer = pints.MultiOutputProblem(model=model_segments, times=times,
                                             values=values_to_match_output_ode)
    ## associate the cost with it
    # error_outer = OuterCriterion(problem=problem_outer)
    error_outer = OuterCriterion(problem=problem_outer)
    init_betas = 0.5 * np.ones(nBsplineCoeffs) # initial values of B-spline coefficients
    sigma0_betas = 0.2 * np.ones(nBsplineCoeffs)
    tic = tm.time()
    model_bsplines = bsplineOutput()
    ## create the problem of comparing the modelled current with measured current
    values_to_match_output_dims = np.transpose(np.array([current_true, voltage, state_known]))
    #^ we actually only need first two columns in this array but pints wants to have the same number of values and outputs
    problem_inner = pints.MultiOutputProblem(model=model_bsplines, times=times, values=values_to_match_output_dims)
     ## associate the cost with it
    error_inner = InnerCriterion(problem=problem_inner)
    ##  define boundaries for the inner optimisation
    boundaries_betas = pints.RectangularBoundaries(np.zeros_like(init_betas), np.ones_like(init_betas))
    ## define boundaries for the outer optimisation
    ####################################################################################################################
    # fit states at the true ODE param values to get the baseline value
    Thetas_ODE = theta_true.copy()
    # fit the b-spline surface given the sampled value of the ODE parameter vector
    betas_sample = []
    inner_costs_sample = []
    end_of_roi = []
    state_fitted_roi = {key: [] for key in state_names}
    deriv_fitted_roi = {key: [] for key in state_names}
    rhs_fitted_roi = {key: [] for key in state_names}
    tic_sample = tm.time()
    for iSegment in range(1):
        segment = times_roi[iSegment]
        input_segment = voltage_roi[iSegment]
        output_segment = current_roi[iSegment]
        support_segment = knots_roi[iSegment]
        state_known_segment = states_known_roi[iSegment]
        # initialise inner optimisation
        init_betas = init_betas_roi[iSegment]
        sigma0_betas = 0.2 * np.ones(nBsplineCoeffs)
        try:
            betas_segment, inner_cost_segment, evals_segment = optimise_first_segment(segment,
                                                                                      input_segment,
                                                                                      output_segment,
                                                                                      support_segment,
                                                                                      state_known_segment)
        except:
            print('Error encountered during opptimisation.')
            optimisationFailed = True
            break  # segments
        # check collocation solution against truth
        knots = support_segment
        model_output = model_bsplines_test.simulate(betas_segment, segment)
        state_at_estimate, deriv_at_estimate, rhs_at_estimate = np.split(model_output, 3, axis=1)
        # add all costs and performance metrics to store for the run
        betas_sample.append(betas_segment)
        inner_costs_sample.append(inner_cost_segment)
        # save the final value of the segment
        end_of_roi.append(state_at_estimate[-1, :])
        for iState, stateName in enumerate(hidden_state_names):
            state_fitted_roi[stateName] += list(state_at_estimate[:, iState])
            # deriv_fitted_roi[stateName] += list(deriv_at_estimate[:, iState])
            # rhs_fitted_roi[stateName] += list(rhs_at_estimate[:,iState])
    ####################################################################################################################
    #  optimise the following segments by matching the first B-spline height to the previous segment
    for iSegment in range(1, len(times_roi)):
        segment = times_roi[iSegment]
        input_segment = voltage_roi[iSegment]
        output_segment = current_roi[iSegment]
        support_segment = knots_roi[iSegment]
        collocation_segment = collocation_roi[iSegment]
        state_known_segment = states_known_roi[iSegment]
        # find the scaling coeff of the first height by matiching its height at t0 of the segment to the final value of the previous segment
        first_spline_coeff = end_of_roi[-1] / collocation_segment[0, 0]
        # initialise inner optimisation
        # we must re-initalise the optimisation with that excludes the first coefficient
        init_betas = init_betas_roi[iSegment][1:]
        sigma0_betas = 0.2 * np.ones(nBsplineCoeffs - 1)  # inital spread of values
        try:
            betas_segment, inner_cost_segment, evals_segment = optimise_segment(segment, input_segment,
                                                                                output_segment,
                                                                                support_segment,
                                                                                state_known_segment)
        except:
            print('Error encountered during opptimisation.')
            optimisationFailed = True
            break  # segments
        # check collocation solution against truth
        knots = support_segment
        model_output = model_bsplines_test.simulate(betas_segment, segment)
        state_at_estimate, deriv_at_estimate, rhs_at_estimate = np.split(model_output, 3, axis=1)
        # add all costs and performance metrics to store for the run
        betas_sample.append(betas_segment)
        inner_costs_sample.append(inner_cost_segment)
        # store end of segment and the whole state for the
        end_of_roi.append(state_at_estimate[-1, :])
        for iState, stateName in enumerate(hidden_state_names):
            state_fitted_roi[stateName] += list(state_at_estimate[1:, iState])
            # deriv_fitted_roi[stateName] += list(deriv_at_estimate[1:, iState])
            # rhs_fitted_roi[stateName] += list(rhs_at_estimate[1:, iState])
    state_all_segments = np.array(state_fitted_roi[hidden_state_names])
    # deriv_all_segments = np.array(deriv_fitted_roi[hidden_state_names])
    # rhs_all_segments = np.array(rhs_fitted_roi[hidden_state_names])
    #### end of loop over segments
    # evaluate the cost functions at the sampled value of ODE parameter vector
    InnerCost_given_true_theta = sum(inner_costs_sample)
    OuterCost_given_true_theta = error_outer(Thetas_ODE)
    # store gradient matching cost just to track evolution
    # d_deriv = (deriv_all_segments - rhs_all_segments) ** 2
    # integral_quad = sp.integrate.simpson(y=d_deriv, even='avg', axis=0)
    # GradCost_given_true_theta = np.sum(integral_quad, axis=0)
    GradCost_given_true_theta = (InnerCost_given_true_theta - OuterCost_given_true_theta)/lambd
    ####################################################################################################################
    # take 1: loosely based on ask-tell example from  pints
    convergence_threshold =10e-8
    iter_for_convergence = 20
    # Create an outer optimisation object
    big_tic = tm.time()
    # optimiser_outer = pints.CMAES(x0=init_thetas,sigma0=sigma0_thetas, boundaries=boundaries_thetas) # with simple rectangular boundaries
    optimiser_outer = pints.CMAES(x0=init_thetas, sigma0=sigma0_thetas, boundaries=boundaries_thetas_Michael) # with boundaries accounting for the reaction rates
    optimiser_outer.set_population_size(min(len(theta_true)*5,30))
    ## Create placeholders for the optimisation
    theta_visited = []
    theta_guessed = []
    f_guessed = []
    theta_best = []
    f_outer_best = []
    f_inner_best = []
    f_gradient_best = []
    InnerCosts_all = []
    OuterCosts_all = []
    GradCost_all = []
    # run outer optimisation for some iterations
    # create a logger file
    csv_file_name = 'iterations_one_state_'+state_name+'.csv'
    column_names = ['Iteration','Walker','Theta_1','Theta_2','Theta_3','Theta_4','Inner Cost', 'Outer Cost', 'Gradient Cost']
    # Create or open the CSV file
    with open(csv_file_name, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(column_names)
        #  run the optimisation
        for i in range(200):
            # get the next points (multiple locations)
            thetas = optimiser_outer.ask()
            # create the placeholder for cost functions
            OuterCosts = []
            InnerCosts = []
            GradCosts = []
            betas_visited = []
            # for each theta in the sample
            tic = tm.time()
            for iTheta, theta in enumerate(thetas):
                # assign the variable that is readable in the class of B-spline evaluation
                Thetas_ODE = theta.copy()
                # fit the b-spline surface given the sampled value of the ODE parameter vector
                betas_sample = []
                inner_costs_sample = []
                end_of_roi = []
                state_fitted_roi = {key: [] for key in state_names}
                # deriv_fitted_roi = {key: [] for key in state_names}
                # rhs_fitted_roi = {key: [] for key in state_names}
                tic_sample = tm.time()
                for iSegment in range(1):
                    segment = times_roi[iSegment]
                    input_segment = voltage_roi[iSegment]
                    output_segment = current_roi[iSegment]
                    support_segment = knots_roi[iSegment]
                    state_known_segment = states_known_roi[iSegment]
                    # initialise inner optimisation
                    init_betas = init_betas_roi[iSegment]
                    sigma0_betas = 0.2 * np.ones(nBsplineCoeffs)
                    try:
                        betas_segment, inner_cost_segment, evals_segment = optimise_first_segment(segment,
                                                                                                  input_segment,
                                                                                                  output_segment,
                                                                                                  support_segment,
                                                                                                  state_known_segment)
                    except:
                        print('Error encountered during opptimisation.')
                        optimisationFailed = True
                        break  # segments
                    # check collocation solution against truth
                    knots = support_segment
                    model_output = model_bsplines_test.simulate(betas_segment, segment)
                    state_at_estimate, deriv_at_estimate, rhs_at_estimate = np.split(model_output, 3, axis=1)
                    # add all costs and performance metrics to store for the run
                    betas_sample.append(betas_segment)
                    inner_costs_sample.append(inner_cost_segment)
                    # save the final value of the segment
                    end_of_roi.append(state_at_estimate[-1, :])
                    for iState, stateName in enumerate(hidden_state_names):
                        state_fitted_roi[stateName] += list(state_at_estimate[:, iState])
                        # deriv_fitted_roi[stateName] += list(deriv_at_estimate[:, iState])
                        # rhs_fitted_roi[stateName] += list(rhs_at_estimate[:, iState])
                ####################################################################################################################
                #  optimise the following segments by matching the first B-spline height to the previous segment
                for iSegment in range(1, len(times_roi)):
                    segment = times_roi[iSegment]
                    input_segment = voltage_roi[iSegment]
                    output_segment = current_roi[iSegment]
                    support_segment = knots_roi[iSegment]
                    collocation_segment = collocation_roi[iSegment]
                    state_known_segment = states_known_roi[iSegment]
                    # find the scaling coeff of the first height by matiching its height at t0 of the segment to the final value of the previous segment
                    first_spline_coeff = end_of_roi[-1] / collocation_segment[0, 0]
                    # initialise inner optimisation
                    # we must re-initalise the optimisation with that excludes the first coefficient
                    init_betas = init_betas_roi[iSegment][1:]
                    sigma0_betas = 0.2 * np.ones(nBsplineCoeffs - 1)  # inital spread of values
                    try:
                        betas_segment, inner_cost_segment, evals_segment = optimise_segment(segment, input_segment,
                                                                                            output_segment,
                                                                                            support_segment,
                                                                                            state_known_segment)
                    except:
                        print('Error encountered during opptimisation.')
                        optimisationFailed = True
                        break  # segments
                    # check collocation solution against truth
                    knots = support_segment
                    model_output = model_bsplines_test.simulate(betas_segment, segment)
                    state_at_estimate, deriv_at_estimate, rhs_at_estimate = np.split(model_output, 3, axis=1)
                    # add all costs and performance metrics to store for the run
                    betas_sample.append(betas_segment)
                    inner_costs_sample.append(inner_cost_segment)
                    # store end of segment and the whole state for the
                    end_of_roi.append(state_at_estimate[-1, :])
                    for iState, stateName in enumerate(hidden_state_names):
                        state_fitted_roi[stateName] += list(state_at_estimate[1:, iState])
                        # deriv_fitted_roi[stateName] += list(deriv_at_estimate[1:, iState])
                        # rhs_fitted_roi[stateName] += list(rhs_at_estimate[1:, iState])
                #### end of loop over segments
                state_all_segments = np.array(state_fitted_roi[hidden_state_names])
                # deriv_all_segments = np.array(deriv_fitted_roi[hidden_state_names])
                # rhs_all_segments = np.array(rhs_fitted_roi[hidden_state_names])
                # evaluate the cost functions at the sampled value of ODE parameter vector
                InnerCosts.append(sum(inner_costs_sample)) # summed over segments
                OuterCosts.append(error_outer(Thetas_ODE))
                # compute gradient matching cost just to track evolution - alternatively can compute just as a difference between inner and outer!
                # d_deriv = (deriv_all_segments - rhs_all_segments) ** 2
                # integral_quad = sp.integrate.simpson(y=d_deriv, even='avg', axis=0)
                # GradCosts.append(np.sum(integral_quad, axis=0))
                GradCosts.append((InnerCosts[-1]-OuterCosts[-1])/lambd) # for now compute simply as the difference between inner and outer to save time
                # store all betas for all segments to get reinitialise fot the following iteration
                betas_visited.append(betas_sample)
                del Thetas_ODE # make sure this is updated
                print(str(iTheta) + '-th sample finished')
                # write to the logger file
                row_to_write = [i] + [iTheta] + list(theta) + [InnerCosts[-1], OuterCosts[-1], GradCosts[-1]]
                writer.writerow(row_to_write)
                del state_all_segments
            ## end loop over samples in the CMA-ES population
            # feed the evaluated scores into the optimisation object
            optimiser_outer.tell(OuterCosts)
            toc = tm.time()
            print(str(i) + '-th iteration finished. Elapsed time: ' + str(toc-tic) + 's')
            # store all costs in the lists
            InnerCosts_all.append(InnerCosts)
            OuterCosts_all.append(OuterCosts)
            GradCost_all.append(GradCosts)
            # HOW DO I CHECK CONVERGENCE HERE - for all points of average cost???
            # Store the requested points
            theta_visited.extend(thetas)
            # # Store the current guess
            # theta_g =np.mean(thetas, axis=0)
            # theta_guessed.append(theta_g)
            # f_guessed.append(error_outer(theta_g))
            # Store the accompanying score
            # Store the best position and score seen so far
            index_best = OuterCosts.index(min(OuterCosts))
            theta_best.append(thetas[index_best,:])
            init_betas_roi = betas_visited[index_best]
            f_outer_best.append(OuterCosts[index_best])
            f_inner_best.append(InnerCosts[index_best])
            f_gradient_best.append(GradCosts[index_best])
            # the most basic convergence condition after running first fifty
            if (i > iter_for_convergence):
                # check how the cost increment changed over the last 10 iterations
                d_cost = np.diff(f_outer_best[-iter_for_convergence:])
                # if all incrementa are below a threshold break the loop
                if all(d<=convergence_threshold for d in d_cost):
                    print("No changes in" + str(iter_for_convergence) + "iterations. Terminating")
                    break
            ## end convergence check condition
        ## end loop over iterations
    ## end of writing to file
    # convert lists into arrays
    theta_visited = np.array(theta_visited)
    # theta_guessed = np.array(theta_guessed)
    theta_best = np.array(theta_best)
    f_outer_best = np.array(f_outer_best)
    f_inner_best = np.array(f_inner_best)
    f_gradient_best = np.array(f_gradient_best)
    # f_guessed = np.array(f_guessed)
    big_toc = tm.time()
    print('Optimisation finished. Elapsed time: ' + str(big_toc-big_tic) + 's')
    n_walkers = int(theta_visited.shape[0] / len(theta_best))
    ####################################################################################################################
    ## write things to files
    column_names = []
    for i in range(n_walkers):
        column_names.append(str(i))
    ## placeholder for storing all visited betas - not sure whether needed
    ####################################################################################################################
    # plot evolution of inner costs
    plt.figure(figsize=(10, 6))
    plt.semilogy()
    plt.xlabel('Iteration')
    plt.ylabel('Inner optimisation cost')
    for iter in range(len(f_outer_best)-1):
        plt.scatter(iter*np.ones(len(InnerCosts_all[iter])),InnerCosts_all[iter], c='k',marker='.', alpha=.5, linewidths=0)
    iter += 1
    plt.scatter(iter * np.ones(len(InnerCosts_all[iter])), InnerCosts_all[iter], c='k', marker='.', alpha=.5,
                linewidths=0,label=r'Sample cost min: $J(C \mid \Theta, \bar{\mathbf{y}}) = $'  +"{:.5e}".format(min(InnerCosts_all[iter])) )
    plt.plot(f_inner_best, '-b', linewidth=1.5,
             label=r'Best cost:$J(C \mid \Theta_{best}, \bar{\mathbf{y}}) = $' + "{:.5e}".format(
                 f_inner_best[-1]))
    plt.plot(range(iter), np.ones(iter) * InnerCost_given_true_theta, '--b', linewidth=2.5, alpha=.5, label=r'Collocation solution: $J(C \mid \Theta_{true}, \bar{\mathbf{y}}) = $'  +"{:.5e}".format(InnerCost_given_true_theta))
    plt.legend(loc='best')
    plt.tight_layout()
    plt.savefig('Figures/inner_cost_ask_tell_one_state_'+stateName+'.png',dpi=400)

    # plot evolution of outer costs
    plt.figure(figsize=(10, 6))
    plt.semilogy()
    plt.xlabel('Iteration')
    plt.ylabel('Outer optimisation cost')
    for iter in range(len(f_outer_best) - 1):
        plt.scatter(iter * np.ones(len(OuterCosts_all[iter])), OuterCosts_all[iter], c='k', marker='.', alpha=.5,
                    linewidths=0)
    iter += 1
    plt.scatter(iter * np.ones(len(OuterCosts_all[iter])), OuterCosts_all[iter], c='k', marker='.', alpha=.5,linewidths=0, label=r'Sample cost: $H(\Theta \mid \hat{C}, \bar{\mathbf{y}})$')
    # plt.plot(range(iter), np.ones(iter) * OuterCost_true, '-m', linewidth=2.5, alpha=.5,label=r'B-splines fit to true state: $H(\Theta \mid  \hat{C}_{direct}, \bar{\mathbf{y}}) = $' + "{:.7f}".format(
    #              OuterCost_true))
    plt.plot(range(iter), np.ones(iter) * OuterCost_given_true_theta, '--m', linewidth=2.5, alpha=.5,label=r'Collocation solution: $H(\Theta_{true} \mid  \hat{C}, \bar{\mathbf{y}}) = $' + "{:.5e}".format(
                 OuterCost_given_true_theta))
    plt.plot(f_outer_best,'-b',linewidth=1.5,label=r'Best cost:$H(\Theta_{best} \mid  \hat{C}, \bar{\mathbf{y}}) = $' + "{:.5e}".format(f_outer_best[-1]))
    plt.legend(loc='best')
    plt.tight_layout()
    plt.savefig('Figures/outer_cost_ask_tell_one_state_'+state_name+'.png',dpi=400)

    # plot evolution of outer costs
    plt.figure(figsize=(10, 6))
    plt.semilogy()
    plt.xlabel('Iteration')
    plt.ylabel('Gradient matching cost')
    for iter in range(len(f_gradient_best) - 1):
        plt.scatter(iter * np.ones(len(GradCost_all[iter])), GradCost_all[iter], c='k', marker='.', alpha=.5,
                    linewidths=0)
    iter += 1
    plt.scatter(iter * np.ones(len(GradCost_all[iter])), GradCost_all[iter], c='k', marker='.', alpha=.5,linewidths=0, label=r'Sample cost: $G_{ODE}(\hat{C}  \mid \Theta, \bar{\mathbf{y}})$')
    plt.plot(range(iter), np.ones(iter) * GradCost_given_true_theta, '--m', linewidth=2.5, alpha=.5,label=r'Collocation solution: $G_{ODE}( \hat{C} \mid  \Theta_{true}, \bar{\mathbf{y}}) = $' + "{:.5e}".format(
                 GradCost_given_true_theta))
    plt.plot(f_gradient_best,'-b',linewidth=1.5,label=r'Best cost:$G_{ODE}(\hat{C} \mid \Theta_{best}, \bar{\mathbf{y}}) = $' + "{:.5e}".format(f_gradient_best[-1]))
    plt.legend(loc='best')
    plt.tight_layout()
    plt.savefig('Figures/gradient_cost_ask_tell_one_state_'+state_name+'.png',dpi=400)

    # plot parameter values after search was done on decimal scale
    fig, axes = plt.subplots(len(theta_true), 1, figsize=(5*len(theta_true), 8), sharex=True)
    for iAx, ax in enumerate(axes.flatten()):
        for iter in range(len(theta_best)):
            x_visited_iter = theta_visited[iter*n_walkers:(iter+1)*n_walkers,iAx]
            ax.scatter(iter*np.ones(len(x_visited_iter)),x_visited_iter,c='k',marker='.',alpha=.2,linewidth=0)
        ax.plot(range(iter+1),np.ones(iter+1)*theta_true[iAx], '--m', linewidth=2.5,alpha=.5, label=r"true: $log("+param_names[iAx]+") = $" +"{:.6f}".format(theta_true[iAx]))
        # ax.plot(theta_guessed[:,iAx],'--r',linewidth=1.5,label=r"guessed: $\theta_{"+str(iAx+1)+"} = $" +"{:.4f}".format(theta_guessed[-1,iAx]))
        ax.plot(theta_best[:,iAx],'-b',linewidth=1.5,label=r"best: $log("+param_names[iAx]+") = $" +"{:.6f}".format(theta_best[-1,iAx]))
        ax.set_ylabel('$log('+param_names[iAx]+')$')
        ax.legend(loc='best')
    plt.tight_layout()
    plt.savefig('Figures/ODE_params_one_state_'+state_name+'.png',dpi=400)

    # plot parameter values converting from log scale to decimal
    fig, axes = plt.subplots(len(theta_true), 1, figsize=(5*len(theta_true), 8), sharex=True)
    n_walkers = int(theta_visited.shape[0] / len(theta_best))
    for iAx, ax in enumerate(axes.flatten()):
        for iter in range(len(theta_best)):
            x_visited_iter = theta_visited[iter*n_walkers:(iter+1)*n_walkers,iAx]
            ax.scatter(iter*np.ones(len(x_visited_iter)),np.exp(x_visited_iter),c='k',marker='.',alpha=.2,linewidth=0)
        ax.plot(range(iter+1),np.ones(iter+1)*np.exp(theta_true[iAx]), '--m', linewidth=2.5,alpha=.5, label="true: $"+param_names[iAx]+" = $" +"{:.6f}".format(np.exp(theta_true[iAx])))
        # ax.plot(np.exp(theta_guessed[:,iAx]),'--r',linewidth=1.5,label="guessed: $a_{"+str(iAx+1)+"} = $" +"{:.4f}".format(np.exp(theta_guessed[-1,iAx])))
        ax.plot(np.exp(theta_best[:,iAx]),'-b',linewidth=1.5,label="best: $"+param_names[iAx]+" = $" +"{:.6f}".format(np.exp(theta_best[-1,iAx])))
        ax.set_ylabel('$'+param_names[iAx]+'$')
        ax.set_yscale('log')
        ax.legend(loc='best')
    ax.set_xlabel('Iteration')
    plt.tight_layout()
    plt.savefig('Figures/ODE_params_one_state_log_scale_'+state_name+'.png',dpi=400)
    ####################################################################################################################
    print('pause here')
    # plot optimised model output
    Thetas_ODE = theta_best[-1]
    state_fitted_roi = {key: [] for key in state_names}
    deriv_fitted_roi = {key: [] for key in state_names}
    rhs_fitted_roi = {key: [] for key in state_names}
    for iSegment in range(1):
        segment = times_roi[iSegment]
        knots = knots_roi[iSegment]
        betas_segment = init_betas_roi[iSegment]
        model_output = model_bsplines_test.simulate(betas_segment, segment)
        state_at_estimate, deriv_at_estimate, rhs_at_estimate = np.split(model_output, 3, axis=1)
        for iState, stateName in enumerate(hidden_state_names):
            state_fitted_roi[stateName] += list(state_at_estimate[:, iState])
            deriv_fitted_roi[stateName] += list(deriv_at_estimate[:, iState])
            rhs_fitted_roi[stateName] += list(rhs_at_estimate[:, iState])
    ####################################################################################################################
    #  optimise the following segments by matching the first B-spline height to the previous segment
    for iSegment in range(1, len(times_roi)):
        segment = times_roi[iSegment]
        knots = knots_roi[iSegment]
        betas_segment = init_betas_roi[iSegment]
        model_output = model_bsplines_test.simulate(betas_segment, segment)
        state_at_estimate, deriv_at_estimate, rhs_at_estimate = np.split(model_output, 3, axis=1)
        for iState, stateName in enumerate(hidden_state_names):
            state_fitted_roi[stateName] += list(state_at_estimate[1:, iState])
            deriv_fitted_roi[stateName] += list(deriv_at_estimate[1:, iState])
            rhs_fitted_roi[stateName] += list(rhs_at_estimate[1:, iState])
    state_all_segments = np.array(state_fitted_roi[hidden_state_names])
    deriv_all_segments = np.array(deriv_fitted_roi[hidden_state_names])
    rhs_all_segments = np.array(rhs_fitted_roi[hidden_state_names])
    current_model = g * state_all_segments[:] * state_known * (voltage - EK)
    fig, axes = plt.subplots(3,1,figsize=(14,9),sharex=True)
    y_labels = ['I', '$\dot{' + state_name + '}$', '$' + state_name + '$']
    axes[0].plot(times,current_true, '-k', label='Current true')
    axes[0].plot(times,current_model, '--r', label='Optimised model output')
    axes[1].plot(times,deriv_all_segments, '--r', label='B-spline derivative')
    axes[1].plot(times, rhs_all_segments, '--c', label='RHS at collocation solution')
    axes[2].plot(times, state_hidden_true, '-k', label='$'+ state_name +'$ true')
    axes[2].plot(times, state_all_segments, '--r', label='Collocation solution')
    for iAx, ax in enumerate(axes.flatten()):
        ax.legend(fontsize=12, loc='best')
        ax.set_ylabel(y_labels[iAx],fontsize=12)
    ax.set_xlabel('time,ms', fontsize=12)
    plt.tight_layout(pad=0.3)
    plt.savefig('Figures/cost_terms_ask_tell_one_state_'+state_name+'.png',dpi=400)

    fig, axes = plt.subplots(3, 1, figsize=(14, 9), sharex=True)
    y_labels = ['$I_{true} - I_{model}$', '$\dot{' + state_name + r'}$ - RHS(\beta)', '$' + state_name + r'$ - \Phi\beta']
    axes[0].plot(times, current_true - current_model, '-k', label='Data error')
    axes[1].plot(times, deriv_all_segments - rhs_all_segments, '--r', label='Derivative error')
    axes[2].plot(times, state_hidden_true - state_all_segments, '-k', label='Approximation error')
    for iAx, ax in enumerate(axes.flatten()):
        ax.legend(fontsize=12, loc='best')
        ax.set_ylabel(y_labels[iAx], fontsize=12)
    ax.set_xlabel('time,ms', fontsize=12)
    plt.tight_layout(pad=0.3)
    plt.savefig('Figures/erros_ask_tell_one_state_' + state_name + '.png', dpi=400)
    ####################################################################################################################
