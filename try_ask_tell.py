# imports
import matplotlib.pyplot as plt

from setup import *
import pints
import pickle as pkl
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
            dadr = ion_channel_model(times, [fun_a, fun_r], Thetas_ODE)
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
            *ps, g = Thetas_ODE[:9]
            volts_for_model = self._values[:,1] # we need to make sure that voltage is read at the times within ROI so we pass it in as part of values
            d_y = g * x[:, 0] * x[:, 1] * (volts_for_model - EK) - self._values[:,0]
            data_fit_cost = np.transpose(d_y) @ d_y
            # compute the gradient matching cost
            d_deriv = (x_dot - rhs) ** 2
            integral_quad = sp.integrate.simpson(y=d_deriv, even='avg',axis=0)
            gradient_match_cost = np.sum(integral_quad, axis=0)
            # not the most elegant implementation because it just grabs global lambda
            return data_fit_cost + lambd * gradient_match_cost

    # define a class that outputs only b-spline surface features
    class ODEOutput(pints.ForwardModel):
        # this model outputs the discrepancy to be used in a rectangle quadrature scheme
        def simulate(self, parameters, times):
            # given times and return the simulated values
            coeffs_a, coeffs_r = np.split(Betas_BSPL, 2)
            tck_a = (knots, coeffs_a, degree)
            tck_r = (knots, coeffs_r, degree)
            dot_a = sp.interpolate.splev(times, tck_a, der=1)
            dot_r = sp.interpolate.splev(times, tck_r, der=1)
            fun_a = sp.interpolate.splev(times, tck_a, der=0)
            fun_r = sp.interpolate.splev(times, tck_r, der=0)
            # the RHS must be put into an array
            dadr = ion_channel_model(times, [fun_a, fun_r], parameters) # this is what depends on the unkonw parameters now!
            rhs_theta = np.array(dadr)
            spline_surface = np.array([fun_a, fun_r])
            spline_deriv = np.array([dot_a, dot_r])
            # pack all required variables into the same array - will be the wrong orientation from pints preferred nTimes x nOutputs
            packed_output = np.concatenate((spline_surface, spline_deriv, rhs_theta), axis=0)
            return np.transpose(packed_output)

        def n_parameters(self):
            # Return the dimension of the parameter vector
            return nBsplineCoeffs

        def n_outputs(self):
            # Return the dimension of the output vector
            return nOutputs

    # define an error w.r.t ODE parameters that assumes that it knows B-spline parameters
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
            # read the ode coefficients from input
            *ps, g = thetas[:9]
            # evaluate the integral at the value of ODE parameters
            model_output = self._problem.evaluate(thetas)   # the output of the model with be an array of size nTimes x nOutputs
            x, x_dot, rhs = np.split(model_output, 3, axis=1) # we split the array into states, state derivs, and RHSs
            # compute the data fit
            volts_for_model = self._values[:,1] # we need to make sure that voltage is read at the times within ROI so we pass it in as part of values
            d_y = g * x[:, 0] * x[:, 1] * (volts_for_model - EK) - self._values[:,0] # this part depends on theta_g
            data_fit_cost = np.transpose(d_y) @ d_y
            # compute the gradient matching cost
            d_deriv = (x_dot - rhs) ** 2
            integral_quad = sp.integrate.simpson(y=d_deriv, even='avg',axis=0)
            gradient_match_cost = np.sum(integral_quad, axis=0)
            # not the most elegant implementation because it just grabs global lambda
            return data_fit_cost + lambd * gradient_match_cost
    ####################################################################################################################
    ## Run the optimisation
    lambd = 0.5 * 10e5
    init_thetas = [2.26e-4, 0.0599, 3.45e-5, 0.04462, 0.0973, 8.91e-3, 5.15e-3, 0.05158, 0.2524]
    init_betas = 0.1 * np.ones(nBsplineCoeffs)  # initial values of B-spline coefficients
    tic = tm.time()
    model_bsplines = bsplineOutput()
    model_ode = ODEOutput()
    ## create the problem of comparing the modelled current with measured current
    voltage = V(times_roi) # must read voltage at the correct times to match the output
    values_to_match_output_dims = np.transpose(np.array([current,voltage,current,voltage,current,voltage]))
    #^ we actually only need first two columns in this array but pints wants to have the same number of values and outputs
    problem_inner = pints.MultiOutputProblem(model=model_bsplines, times=times_roi, values=values_to_match_output_dims)
    problem_outer = pints.MultiOutputProblem(model=model_ode, times=times_roi, values=values_to_match_output_dims)
    ## associate the cost with it
    error_inner  = InnerCriterion(problem=problem_inner)
    error_outer = OuterCriterion(problem=problem_outer)
    ##  define boundaries for the inner optimisation
    boundaries_betas = pints.RectangularBoundaries(np.zeros_like(init_betas),0.6 * np.ones_like(init_betas))
    ## define boundaries for the outer optimisation
    boundaries_thetas = pints.RectangularBoundaries(np.zeros_like(init_thetas), np.ones_like(init_thetas))
    ####################################################################################################################
    ## get error measures at true ODE parameter values
    optimiser_inner = pints.OptimisationController(error_inner, x0=init_betas, boundaries=boundaries_betas,
                                                   method=pints.CMAES)
    optimiser_inner.set_max_iterations(30000)
    optimiser_inner.set_max_unchanged_iterations(iterations=50, threshold=1e-6)
    optimiser_inner.set_parallel(False)
    optimiser_inner.set_log_to_screen(True)
    Betas_BSPL, InnerCost_true = optimiser_inner.run()
    OuterCost_true = error_outer(p_true)
    Betas_given_true = Betas_BSPL.copy()
    ####################################################################################################################
    # take 1: loosely based on ask-tel example from  pints
    convergence_threshold = 1e-6
    # Create an outer optimisation object
    big_tic = tm.time()
    optimiser_outer = pints.CMAES(x0=init_thetas, boundaries=boundaries_thetas)
    ## Run optimisation
    x_visited = []
    x_guessed = []
    x_best = []
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
            # fit the b-spline surface given the sampled value of ODE parameter vector
            # introduce an optimiser every time beacause it does not understand why thre is already an instance of the optimier
            optimiser_inner = pints.OptimisationController(error_inner, x0=init_betas, boundaries=boundaries_betas,
                                                           method=pints.CMAES)
            optimiser_inner.set_max_iterations(30000)
            optimiser_inner.set_max_unchanged_iterations(iterations=50, threshold=1e-6)
            optimiser_inner.set_parallel(False)
            optimiser_inner.set_log_to_screen(False)
            Betas_BSPL, InnerCost = optimiser_inner.run()
            # init_betas = Betas_BSPL # update the init conds for the next itimiser instance
            # evaluate the cost function at the sampled value of ODE parameter vector
            InnerCosts.append(InnerCost)
            OuterCosts.append(error_outer(theta))
            del Thetas_ODE # bin this variable to make sure it is not updates somewhere else
        # feed the evaluated scores into the optimisation object
        optimiser_outer.tell(OuterCosts)
        toc = tm.time()
        print(str(i) + '-th iteration finished. Elapsed time: ' + str(toc-tic) + 's')
        # store all costs in the lists
        InnerCosts_all.append(InnerCosts)
        OuterCosts_all.append(OuterCosts)
        # HOW DO I CHECK CONVERGENCE HERE - for all points of average cost???
        # Store the requested points
        x_visited.extend(thetas)
        # Store the current guess
        x_guessed.append(np.mean(thetas, axis=0))
        # Store the accompanying score
        # Store the best position and score seen so far
        x_best.append(optimiser_outer._xbest)
        f_best.append(optimiser_outer._fbest)
        # the most basic convergence condition after running first fifty
        if (iter >50):
            # check how the cost increment changed over the last 10 iterations
            d_cost = np.diff(f_best[-10:])
            # if all incrementa are below a threshold break the loop
            if all(d<=convergence_threshold for d in d_cost):
                break
    # convert lists into arrays
    x_visited = np.array(x_visited)
    x_guessed = np.array(x_guessed)
    x_best = np.array(x_best)
    f_best = np.array(f_best)
    big_toc = tm.time()
    print('Optimisation finished. Elapsed time: ' + str(big_toc-big_tic) + 's')
    ####################################################################################################################
    # try saving the results
    results_to_save = [InnerCosts_all,OuterCosts_all,x_best,f_best,x_visited,x_guessed]
    with open("ask_tell_iterations.pkl", "wb") as output_file:
        pkl.dump(results_to_save, output_file)
    ####################################################################################################################
    # plot evolution of outer costs
    plt.figure()
    plt.semilogy()
    plt.xlabel('Iteration')
    plt.ylabel('Outer optimisation cost')
    for iter in range(len(f_best)):
        plt.scatter(iter * np.ones(len(InnerCosts_all[iter])), InnerCosts_all[iter], c='k', marker='.', alpha=.5,
                    linewidths=0)
    plt.plot(range(iter), np.ones(iter) * InnerCost_true, '-m', linewidth=2.5, alpha=.5, label='Cost at truth')
    plt.legend(loc='best')
    plt.tight_layout()
    plt.savefig('Figures/inner_cost_ask_tell_dif_costs.png')

    # plot evolution of inner costs
    plt.figure()
    plt.semilogy()
    plt.xlabel('Iteration')
    plt.ylabel('Inner optimisation cost')
    for iter in range(len(f_best)):
        plt.scatter(iter * np.ones(len(OuterCosts_all[iter])), OuterCosts_all[iter], c='k', marker='.', alpha=.5,
                    linewidths=0)
    plt.plot(range(iter), np.ones(iter) * OuterCost_true, '-m', linewidth=2.5, alpha=.5, label='Cost at truth')
    plt.plot(f_best, '-b', linewidth=1.5, label='Best cost')
    plt.legend(loc='best')
    plt.tight_layout()
    plt.savefig('Figures/outer_cost_ask_tell_dif_costs.png')

    # plot parameter values
    fig, axes = plt.subplots(3, 3, figsize=(20, 8), sharex=True)
    n_walkers = int(x_visited.shape[0] / len(x_best))
    for iAx, ax in enumerate(axes.flatten()):
        for iter in range(len(x_best)):
            x_visited_iter = x_visited[iter * n_walkers:(iter + 1) * n_walkers, iAx]
            ax.scatter(iter * np.ones(len(x_visited_iter)), x_visited_iter, c='k', marker='.', alpha=.2,
                       linewidth=0)
        ax.plot(range(iter), np.ones(iter) * p_true[iAx], '-m', linewidth=2.5, alpha=.5, label='true')
        ax.plot(x_guessed[:, iAx], '--r', linewidth=1.5, label='guessed')
        ax.plot(x_best[:, iAx], '-b', linewidth=1.5, label='best')
        ax.set_ylabel('$p_{' + str(iAx) + '}$')
    ax.legend(loc='best')
    plt.tight_layout()
    plt.savefig('Figures/ODE_params_ask_tell_dif_costs.png')

    # plot model output
    current_true = observation(times_roi, x_ar, p_true)
    Thetas_ODE = x_best[-1, :]
    optimiser_inner = pints.OptimisationController(error_inner, x0=init_betas, boundaries=boundaries_betas,
                                                   method=pints.CMAES)
    optimiser_inner.set_max_iterations(30000)
    optimiser_inner.set_max_unchanged_iterations(iterations=50, threshold=1e-6)
    optimiser_inner.set_parallel(False)
    optimiser_inner.set_log_to_screen(False)
    Betas_BSPL, BSPL_cost = optimiser_inner.run()
    opt_model_output = model_bsplines.simulate(Betas_BSPL, times_roi)
    states, state_derivs, rhs = np.split(opt_model_output, 3, axis=1)
    *ps, g = Thetas_ODE[:9]
    current_model = g * states[:, 0] * states[:, 1] * (voltage - EK)
    fig, axes = plt.subplots(2, 3)
    y_labels = ['I', '$\dot{a}$', '$\dot{r}$', 'a', 'r']
    axes[0, 2].plot(times_roi, current_true, '-k', label='Current true')
    axes[0, 2].plot(times_roi, current_model, '--r', label='Optimised model output')
    axes[0, 0].plot(times_roi, rhs[:, 0], '-k', label='$\dot{a}$')
    axes[0, 0].plot(times_roi, state_derivs[:, 0], '--r', label='B-spline derivative')
    axes[0, 1].plot(times_roi, rhs[:, 1], '-k', label='$\dot{r}$')
    axes[0, 1].plot(times_roi, state_derivs[:, 1], '--r', label='B-spline derivative')
    axes[1, 0].plot(times_roi, x_ar[0, :], '-k', label=y_labels[3] + ' true')
    axes[1, 0].plot(times_roi, states[:, 0], '--r', label='B-spline approximation')
    axes[1, 1].plot(times_roi, x_ar[1, :], '-k', label=y_labels[4] + ' true')
    axes[1, 1].plot(times_roi, states[:, 1], '--r', label='B-spline approximation')
    for iAx, ax in enumerate(axes.flatten()[:-1]):
        ax.set_xlabel('time,ms', fontsize=12)
        ax.legend(fontsize=12, loc='lower right')
    #     ax.set_ylabel(y_labels[iAx],fontsize=8)
    plt.tight_layout(pad=0.3)
    # plt.ioff()
    plt.savefig('Figures/cost_terms_ask_tell_dif_costs.png')
    ####################################################################################################################
    print('pause here')