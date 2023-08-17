# imports
from setup import *
import pints
import pickle as pkl

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
    tlim = [0, 3600]
    times = np.linspace(*tlim, tlim[-1])
    # define a region of interest - we will need this to preserve the
    # trajectories of states given the full clamp and initial position, while
    ROI_start = 3300
    ROI_end = tlim[-1]
    ROI = range(ROI_start,ROI_end)
    # get time points to compute the fit to ODE cost
    times_roi = times[ROI_start:ROI_end]
    times_quad = np.linspace(times_roi[0], times_roi[-1],num=2*len(ROI)) # set up time nodes for quadrature integration
    volts_new = V(times)
    d2v_dt2 = np.diff(volts_new, n=2)
    switchpoints = np.abs(d2v_dt2) > 1e-6
    # ignore everything outside of the region of iterest
    switchpoints_roi = switchpoints[ROI_start:ROI_end]

    ## Generate the synthetic data
    # parameter values for the model
    EK = -80
    p_true = [2.26e-4, 0.0699, 3.45e-5, 0.05462, 0.0873, 8.91e-3, 5.15e-3, 0.03158, 0.1524]
    # initialise and solve ODE
    x0 = [0, 1]
    # solve initial value problem
    solution = sp.integrate.solve_ivp(ion_channel_model, tlim, x0, args=[p_true], dense_output=True,method='LSODA',rtol=1e-8,atol=1e-8)
    x_ar = solution.sol(times_roi)
    current = observation(times_roi, x_ar, p_true)

    ####################################################################################################################
    ## B-spline representation setup
    # set times of jumps and a B-spline knot sequence
    nPoints_closest = 15  # the number of points from each jump where knots are placed at the finest grid
    nPoints_between_closest = 5  # step between knots at the finest grid
    nPoints_around_jump = 45  # the time period from jump on which we place medium grid
    step_between_knots = 45  # this is the step between knots around the jump in the medium grid
    nPoints_between_jumps = 2  # this is the number of knots at the coarse grid corresponding to slowly changing values

    # get the times of all jumps
    jump_indeces = [0] + [i for i, x in enumerate(switchpoints_roi) if x] + [
        len(ROI)]  # get indeces of all the switchpoints, add t0 and tend
    # jump_indeces =  [i for i, x in enumerate(switchpoints_new) if x] # indeces of switchpoints only
    abs_distance_lists = [[(num - index) for num in range(len(ROI) + 1)] for index in
                          jump_indeces]  # compute absolute distance between each time and time of jump
    min_pos_distances = [min(filter(lambda x: x >= 0, lst)) for lst in zip(*abs_distance_lists)]
    max_neg_distances = [max(filter(lambda x: x <= 0, lst)) for lst in zip(*abs_distance_lists)]
    first_jump_index = np.where(np.array(min_pos_distances) == 0)[0][1]
    min_pos_distances[:first_jump_index] = [np.inf] * len(min_pos_distances[:first_jump_index])
    last_jump_index = np.where(np.array(max_neg_distances) == 0)[0][-2]
    max_neg_distances[last_jump_index:] = [-np.inf] * len(max_neg_distances[last_jump_index:])
    knots_after_jump = [
        ((x <= nPoints_closest) and (x % nPoints_between_closest == 0)) or (
                (nPoints_closest <= x <= nPoints_around_jump) and (x % step_between_knots == 0)) for
        x in min_pos_distances]  # create a knot sequence that has higher density of knots after each jump
    knots_before_jump = [((x >= -nPoints_closest) and (x % (nPoints_closest + 1) == 0)) for x in
                         max_neg_distances]  # list on knots befor each jump
    knots_jump = [a or b for a, b in zip(knots_after_jump, knots_before_jump)]
    # add t0 and t_end as a single point in the end
    knots_jump[0] = True
    knots_jump[-1] = True  # logical sum for two boolean lists
    # to do this we then need to add additional coarse grid of knots between two jumps:
    knot_times = [i + ROI_start for i, x in enumerate(knots_jump) if x]  # convert to numeric array again
    # add the final time point in case it is not already included - we need this if we are only adding values after steps
    if not np.isin(ROI_end, knot_times):
        knot_times.append(ROI_end)
    knots_all = knot_times.copy()
    for iKnot, timeKnot in enumerate(knot_times[:-1]):
        if knot_times[iKnot + 1] - timeKnot > step_between_knots:
            # create evenly spaced points and drop start and end - those are already in the grid
            knots_between_jumps = np.rint(
                np.linspace(timeKnot, knot_times[iKnot + 1], num=nPoints_between_jumps + 2)[1:-1]).astype(int)
            # add indeces to the list
            knots_all = knots_all + list(knots_between_jumps)
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
    # fig, ax = plt.subplots()
    for i in range(len(coeffs)):
        tau_current = np.arange(knots[i], knots[i + 4])
        coeffs[i] = 1
        splinest[i] = BSpline(knots, coeffs.copy(), degree,
                              extrapolate=False)  # create a spline that only has one non-zero coeff
        # ax.plot(tau_current, splinest[i](tau_current), lw=0.5, alpha=0.7)
        coeffs[i] = 0
    collocation = collocm(splinest, tau)  # create a collocation matrix for that interval
    ####################################################################################################################
    ## Classes to run optimisation in pints
    nBsplineCoeffs = len(coeffs) * 2  # this to be used in params method of class ForwardModel
    print('Number of B-spline coeffs: ' + str(nBsplineCoeffs))
    nOutputs = 6
    # define a class that outputs only b-spline surface features
    class bsplineOutput(pints.ForwardModel):
        # this model outputs the discrepancy to be used in a rectangle quadrature scheme
        def simulate(self, parameters, times):
            # given times and return the simulated values
            coeffs_a, coeffs_r = np.split(parameters, 2)
            tck_a = (knots, coeffs_a, degree)
            tck_r = (knots, coeffs_r, degree)
            dot_a = sp.interpolate.splev(times, tck_a, der=1)
            dot_r = sp.interpolate.splev(times, tck_r, der=1)
            fun_a = sp.interpolate.splev(times, tck_a, der=0)
            fun_r = sp.interpolate.splev(times, tck_r, der=0)
            # the RHS must be put into an array
            dadr = ion_channel_model(times, [fun_a, fun_r], p_true)
            rhs_theta = np.array(dadr)
            spline_surface = np.array([fun_a, fun_r])
            spline_deriv = np.array([dot_a, dot_r])
            # pack all required variables into the same array - will be the wrong orientation from pints preferred nTimes x nOutputs
            packed_output = np.concatenate((spline_surface,spline_deriv,rhs_theta),axis=0)
            return np.transpose(packed_output)

        def n_parameters(self):
            # Return the dimension of the parameter vector
            return nBsplineCoeffs

        def n_outputs(self):
            # Return the dimension of the output vector
            return nOutputs

    # define an error that only creates B-spline surface ones and computes a sum of costs
    class CompositeError(pints.ProblemErrorMeasure):
        # do I need to redefine custom init or can just drop this part?
        def __init__(self, problem, weights=None):
            super(CompositeError, self).__init__(problem)
            if weights is None:
                weights = [1] * self._n_outputs
            elif self._n_outputs != len(weights):
                raise ValueError(
                    'Number of weights must match number of problem outputs.')
            # Check weights
            self._weights = np.asarray([float(w) for w in weights])

        def __call__(self, theta):
            # evaluate the integral at the value of parameters x
            model_output = self._problem.evaluate(theta)   # the output of the model with be an array of size nTimes x nOutputs
            x, x_dot, rhs = np.split(model_output, 3, axis=1) # we split the array into states, state derivs, and RHSs
            # compute the data fit
            *ps, g = p_true[:9]
            volts_for_model = self._values[:,1] # we need to make sure that voltage is read at the times within ROI so we pass it in as part of values
            d_y = g * x[:, 0] * x[:, 1] * (volts_for_model - EK) - self._values[:,0]
            data_fit_cost = np.transpose(d_y) @ d_y
            # compute the gradient matching cost
            d_deriv = (x_dot - rhs) ** 2
            integral_quad = sp.integrate.simpson(y=d_deriv, even='avg',axis=0)
            gradient_match_cost = np.sum(integral_quad, axis=0)
            # not the most elegant implementation because it just grabs global lambda
            return data_fit_cost + lambd * gradient_match_cost


    class CustomBoundaries(pints.Boundaries):
        def __init__(self, lower, upper):
            super(CustomBoundaries, self).__init__()

            # Convert to shape (n,) vectors, copy to ensure they remain unchanged
            self._lower = pints.vector(lower)
            self._upper = pints.vector(upper)

            # Get and check dimension
            self._n_parameters = len(self._lower)
            if len(self._upper) != self._n_parameters:
                raise ValueError('Lower and upper bounds must have same length.')

            # Check dimension is at least 1
            if self._n_parameters < 1:
                raise ValueError('The parameter space must have a dimension > 0')

            # Check if upper > lower
            if not np.all(self._upper > self._lower):
                raise ValueError('Upper bounds must exceed lower bounds.')


    def check(self, parameters):
        """ See :meth:`pints.Boundaries.check()`. """
        coeffs_a, coeffs_r = np.split(parameters, 2)
        tck_a = (knots, coeffs_a, degree)
        tck_r = (knots, coeffs_r, degree)
        fun_a = sp.interpolate.splev(times, tck_a, der=0)
        fun_r = sp.interpolate.splev(times, tck_r, der=0)
        if np.any(fun_a < self._lower) or np.any(fun_r < self._lower):
            return False
        if np.any(fun_a >= self._upper) or np.any(fun_r >= self._upper):
            return False
        return True

    def n_parameters(self):
        """ See :meth:`pints.Boundaries.n_parameters()`. """
        return self._n_parameters

    def lower(self):
        """
        Returns the lower boundaries for all parameters (as a read-only NumPy
        array).
        """
        return self._lower

    def upper(self):
        """
        Returns the upper boundary for all parameters (as a read-only NumPy
        array).
        """
        return self._upper

    ####################################################################################################################
    ## Run the optimisation
    lambd = 0.5 * 10e5
    init_spline_betas = 0.1 * np.ones(nBsplineCoeffs)  # initial parameter values
    tic = tm.time()
    model_new = bsplineOutput()
    # create the problem of comparing the modelled current with measured current
    voltage = V(times_roi)
    values_to_match_output_dims = np.transpose(np.array([current,voltage,current,voltage,current,voltage]))
    problem_new = pints.MultiOutputProblem(model=model_new, times=times_roi, values=values_to_match_output_dims)
    # associate the cost with it
    error_new  = CompositeError(problem=problem_new)
    boundariesBetas = pints.RectangularBoundaries(np.zeros_like(init_spline_betas),0.6 * np.ones_like(init_spline_betas))
    optimiser_new = pints.OptimisationController(error_new, x0=init_spline_betas, boundaries=boundariesBetas,method=pints.CMAES)
    optimiser_new.set_max_iterations(30000)
    optimiser_new.set_max_unchanged_iterations(iterations=50, threshold=1e-6)
    optimiser_new.set_log_to_file(filename='New_iterations.csv', csv=True)
    found_parameters, found_value = optimiser_new.run()
    # save found params
    with open("B_spline_coeffs_faster_model.pkl", "wb") as output_file:
        pkl.dump(found_parameters, output_file)
    toc = tm.time()
    print('Time elapsed: ' + str(toc-tic) + 's')
    coeffs_a, coeffs_r = np.split(found_parameters, 2)
    tck_a = (knots, coeffs_a, degree)
    tck_r = (knots, coeffs_r, degree)
    dot_a = sp.interpolate.splev(times_roi, tck_a, der=1)
    dot_r = sp.interpolate.splev(times_roi, tck_r, der=1)
    fun_a = sp.interpolate.splev(times_roi, tck_a, der=0)
    fun_r = sp.interpolate.splev(times_roi, tck_r, der=0)
    colloc_a = coeffs_a @ collocation
    colloc_r = coeffs_r @ collocation
    dadr_all = ion_channel_model(times_roi, [fun_a, fun_r], p_true)
    rhs_theta = np.array(dadr_all)
    toc = tm.time()
    print('Time elapsed: ' + str(toc-tic) + 's')
    ####################################################################################################################
    # plot the output of the optimised model
    fig, axes = plt.subplots(2, 2,figsize=(12,8), sharex=True)
    axes[0,0].plot(times_roi,x_ar[0], '-k', label='true')
    axes[0,0].plot(times_roi,fun_a, '--r', label='B-spline fit')
    # axes[0,0].plot(times_roi,colloc_a, '.b', label='Collocation method')
    axes[0,0].set_ylabel('a')
    axes[1,0].plot(times_roi,x_ar[1], '-k', label='true')
    axes[1,0].plot(times_roi,fun_r, '--r', label='B-spline fit')
    # axes[1,0].plot(times_roi,colloc_r, '.b', label='Collocation method')
    axes[1, 0].set_ylabel('r')
    axes[0,1].plot(times_roi,colloc_a, '--r', label='Smoothed curve')
    coeffs = np.zeros(len(knots) - degree - 1)  # number of splines will depend on the knot order
    splinest = [None] * len(coeffs)  # the grid of indtividual splines is required to generate a collocation matrix
    for i in range(len(coeffs)):
        tau_current = np.arange(knots[i], knots[i + 4])
        coeffs[i] =  coeffs_a[i]
        splinest[i] = BSpline(knots, coeffs.copy(), degree, extrapolate=False)  # create a spline that only has one non-zero coeff
        axes[0,1].plot(tau_current, splinest[i](tau_current), lw=0.5, alpha=0.7)
        coeffs[i] = 0
    axes[1,1].plot(times_roi,colloc_r, '--r', label='Smoothed curve')
    for i in range(len(coeffs)):
        tau_current = np.arange(knots[i], knots[i + 4])
        coeffs[i] = coeffs_r[i]
        splinest[i] = BSpline(knots, coeffs.copy(), degree, extrapolate=False)  # create a spline that only has one non-zero coeff
        axes[1, 1].plot(tau_current, splinest[i](tau_current), lw=0.5, alpha=0.7)
        coeffs[i] = 0
    for iAx, ax in enumerate(axes.flatten()):
        ax.set_xlabel('time,ms')
        ax.legend(fontsize=14, loc='upper right')
    plt.tight_layout()
    plt.savefig('Figures/Opt_b_spline_grids_faster.png')

    opt_model_output = model_new.simulate(found_parameters,times_roi)
    states, state_derivs, rhs = np.split(opt_model_output, 3, axis=1)
    *ps, g = p_true[:9]
    current_model = g * states[:, 0] * states[:, 1] * (voltage - EK)
    fig, axes = plt.subplots(3,1,figsize=(12,8), sharex=True)
    y_labels = ['I', '$\dot{a}$', '$\dot{r}$']
    axes[0].plot(times_roi,current, '-k', label='true')
    axes[0].plot(times_roi,current_model, '--r', label='Optimised model output')
    axes[1].plot(times_roi,rhs[:,0], '-k', label='RHS')
    axes[1].plot(times_roi,state_derivs[:,0], '--r', label='B-spline derivative')
    axes[2].plot(times_roi,rhs[:,1], '-k', label='RHS')
    axes[2].plot(times_roi,state_derivs[:,1], '--r', label='B-spline derivative')
    for iAx, ax in enumerate(axes.flatten()):
        ax.set_xlabel('time,ms')
        ax.legend(fontsize=14, loc='upper right')
        ax.set_ylabel(y_labels[iAx])
    ax.set_xlabel('time,ms')
    plt.tight_layout()
    ax.legend(fontsize=14, loc='upper right')
    plt.savefig('Figures/costs_faster.png')
    ##
    print('pause here')