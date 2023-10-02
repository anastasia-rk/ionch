# imports
import matplotlib.pyplot as plt

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

def almost_block_diag(arrs):
    # this function creates an almost diagonal matrix that accomodates 1 dt overlap where segments join
    # number of rows is the same for each segment
    rr = len(arrs[0])
    # create matrix of zeros
    out = np.zeros([rr*len(arrs), len(times)])
    #  counters
    r = 0
    c = 0
    for i in range(len(arrs)):
        # number of columns will be different beacuse they correspond to time
        cc = len(arrs[i][0])
        # populate matrix with block
        out[r:r + rr, c:c + cc] = arrs[i]
        r+=rr
        c+=cc-1
    return out

def optimise_first_segment(roi,input_roi,output_roi,support_roi):
    class bsplineOutput(pints.ForwardModel):
        # this model outputs the discrepancy to be used in a rectangle quadrature scheme
        def simulate(self, parameters, times):
            # given times and return the simulated values
            coeffs_a, coeffs_r = np.split(parameters, 2)
            tck_a = (support_roi, coeffs_a, degree)
            tck_r = (support_roi, coeffs_r, degree)
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
    # define a class that outputs only b-spline surface features
    model_bsplines = bsplineOutput()
    init_betas = 0.5 * np.ones(nBsplineCoeffs)  # initial values of B-spline coefficients
    sigma0_betas = 0.2 * np.ones(nBsplineCoeffs)
    values_to_match_output_dims = np.transpose( np.array([output_roi, input_roi, output_roi, input_roi, output_roi, input_roi]))
    problem_inner = pints.MultiOutputProblem(model=model_bsplines, times=roi, values=values_to_match_output_dims)
    error_inner = InnerCriterion(problem=problem_inner)
    boundaries_betas = pints.RectangularBoundaries(np.zeros_like(init_betas), 0.99 * np.ones_like(init_betas))
    optimiser_inner = pints.OptimisationController(error_inner, x0=init_betas, sigma0=sigma0_betas,
                                                   boundaries=boundaries_betas, method=pints.CMAES)
    optimiser_inner.set_max_iterations(30000)
    optimiser_inner.method().set_population_size(min(5, len(init_betas)/2))
    optimiser_inner.set_max_unchanged_iterations(iterations=50, threshold=1e-10)
    optimiser_inner.set_parallel(False)
    optimiser_inner.set_log_to_screen(False)
    betas_roi, cost_roi = optimiser_inner.run()
    nEvaluations = optimiser_inner._evaluations
    return betas_roi, cost_roi, nEvaluations

def optimise_segment(roi,input_roi,output_roi,support_roi):
    class bsplineOutputSegment(pints.ForwardModel):
        # this model outputs the discrepancy to be used in a rectangle quadrature scheme
        def simulate(self, parameters, times):
            # given times and return the simulated values
            coeffs_a, coeffs_r = np.split(parameters, 2)
            coeffs_a = np.insert(coeffs_a, 0, first_spline_coeff_a)
            coeffs_r = np.insert(coeffs_r, 0, first_spline_coeff_r)
            tck_a = (support_roi, coeffs_a, degree)
            tck_r = (support_roi, coeffs_r, degree)
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
            return nBsplineCoeffs-2

        def n_outputs(self):
            # Return the dimension of the output vector
            return nOutputs
    # define a class that outputs only b-spline surface features
    model_bsplines = bsplineOutputSegment()
    init_betas = 0.5 * np.ones(nBsplineCoeffs-2)  # initial values of B-spline coefficients
    sigma0_betas = 0.2 * np.ones(nBsplineCoeffs-2)
    values_to_match_output_dims = np.transpose( np.array([output_roi, input_roi, output_roi, input_roi, output_roi, input_roi]))
    problem_inner = pints.MultiOutputProblem(model=model_bsplines, times=roi, values=values_to_match_output_dims)
    error_inner = InnerCriterion(problem=problem_inner)
    boundaries_betas = pints.RectangularBoundaries(np.zeros_like(init_betas), 0.99 * np.ones_like(init_betas))
    optimiser_inner = pints.OptimisationController(error_inner, x0=init_betas, sigma0=sigma0_betas,
                                                   boundaries=boundaries_betas, method=pints.CMAES)
    optimiser_inner.set_max_iterations(30000)
    optimiser_inner.method().set_population_size(min(5, len(init_betas)/2))
    optimiser_inner.set_max_unchanged_iterations(iterations=50, threshold=1e-10)
    optimiser_inner.set_parallel(False)
    optimiser_inner.set_log_to_screen(False)
    betas_roi, cost_roi = optimiser_inner.run()
    nEvaluations = optimiser_inner._evaluations
    coeffs_a, coeffs_r = np.split(betas_roi, 2)
    coeffs_a = np.insert(coeffs_a,0,first_spline_coeff_a)
    coeffs_r = np.insert(coeffs_r,0,first_spline_coeff_r)
    betas_roi_with_first_coeff = np.concatenate((coeffs_a,coeffs_r))
    return betas_roi_with_first_coeff, cost_roi, nEvaluations

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
    tlim = [3500, 6000]
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
    solution = sp.integrate.solve_ivp(ion_channel_model, tlim, x0, args=[thetas_true], dense_output=True,
                                      method='LSODA', rtol=1e-8, atol=1e-8)

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
    # ignore everything outside of the region of iterest
    # get the times of all jumps
    a = [0] + [i for i, x in enumerate(switchpoints) if x] + [len(switchpoints)]  # get indeces of all the switchpoints, add t0 and tend
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
    current_roi = []
    voltage_roi = []
    knots_roi = []
    collocation_roi = []
    colderiv_roi = []
    for iJump, jump in enumerate(jump_indeces[:-1]):
        # define a region of interest - we will need this to preserve the
        # trajectories of states given the full clamp and initial position, while
        ROI_start = jump
        ROI_end = jump_indeces[iJump+1]+1 # add one to ensure that t_end equals to t_start of the following segment
        ROI = times[ROI_start:ROI_end]
        x_ar = solution.sol(ROI)
        # get time points to compute the fit to ODE cost
        times_roi.append(ROI)
        # save states
        states_roi.append(x_ar)
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
            splinest[i] = BSpline(knots, coeffs.copy(), degree,
                                  extrapolate=False)  # create a spline that only has one non-zero coeff
            splineder[i] = splinest[i].derivative(nu=1)
            coeffs[i] = 0.
        collocation_roi.append(collocm(splinest, ROI))
        colderiv_roi.append(collocm(splineder, ROI))# create a collocation matrix for that interval
    ##^ this loop stores the time intervals from which to draw collocation points and the data for piece-wise fitting
    # ####################################################################################################################
    # # evaluate full collocation matrix for the entire time-series - will be needed for outer optimisation
    # collocation_whole = almost_block_diag(collocation_roi)
    # colderiv_whole = almost_block_diag(colderiv_roi)
    # fig, axes = plt.subplots(2,1,figsize=(16,8),sharex=True)
    # for spline_values in collocation_whole:
    #     axes[0].plot(times[:],spline_values,lw=0.5)
    # for i in jump_indeces:
    #     axes[0].axvline(x=times[i],color='black', ls='--', lw=0.5)
    # axes[0].set_ylabel('r$\phi_i(t)$')
    # for deriv_values in colderiv_whole:
    #     axes[1].plot(times[:],deriv_values,lw=0.5)
    # for i in jump_indeces:
    #     axes[1].axvline(x=times[i],color='black', ls='--', lw=0.5)
    # axes[1].set_ylabel('r$\dot{\phi}_i(t)$')
    # axes[1].set_xlabel('t, ms')
    # plt.tight_layout(pad=0.3)
    # plt.savefig('Figures/check_collocation_matrices.png',dpi=400)
    #
    # fig, axes = plt.subplots(2, 1, figsize=(16, 8), sharex=True)
    # for spline_values in collocation_whole:
    #     axes[0].plot(times[:], spline_values, lw=1)
    # for i in jump_indeces[:-1]:
    #     axes[0].axvline(x=times[i], color='black', ls='--', lw=1.5)
    #     axes[0].axvline(x=times[i + nPoints_closest-1], color='purple', ls='--', lw=1)
    #     axes[0].axvline(x=times[i + nPoints_around_jump-1], color='blue', ls='--', lw=1)
    # axes[0].set_ylabel('r$\phi_i(t)$')
    # axes[0].set_xlim(240, 480)
    # for deriv_values in colderiv_whole:
    #     axes[1].plot(times[:], deriv_values, lw=1)
    # for i in jump_indeces[:-1]:
    #     axes[1].axvline(x=times[i], color='black', ls='--', lw=1.5)
    #     axes[1].axvline(x=times[i + nPoints_closest-1], color='purple', ls='--', lw=1)
    #     axes[1].axvline(x=times[i + nPoints_around_jump-1], color='blue', ls='--', lw=1)
    # axes[1].set_ylabel('r$\dot{\phi}_i(t)$')
    # axes[1].set_xlabel('t, ms')
    # axes[1].set_xlim(240, 480)
    # plt.tight_layout(pad=0.3)
    # plt.savefig('Figures/zoom_on_segment.png', dpi=400)
    ####################################################################################################################
    ## define pints classes for optimisation
    ## Classes to run optimisation in pints
    lambd = 10000
    nBsplineCoeffs = len(coeffs) * 2  # number of B-spline coefficients for a segment
    Thetas_ODE = thetas_true.copy() # initial values of the ODE parametes
    nOutputs = 6

    class bsplineOutputTest(pints.ForwardModel):
        # this model outputs the discrepancy to be used in a rectangle quadrature scheme
        def simulate(self, parameters, support, times):
            # given times and return the simulated values
            coeffs_a, coeffs_r = np.split(parameters, 2)
            tck_a = (support, coeffs_a, degree)
            tck_r = (support, coeffs_r, degree)
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
    ####################################################################################################################
    # try optimising several segments
    all_betas = []
    all_costs = []
    *ps, g = thetas_true
    model_bsplines = bsplineOutputTest()
    end_of_roi = []
    state_of_roi = {key: [] for key in state_names}
    totalEvaluations = []
    big_tic = tm.time()
    for iSegment in range(1):
        tic = tm.time()
        segment = times_roi[iSegment]
        input_segment = voltage_roi[iSegment]
        output_segment = current_roi[iSegment]
        support_segment = knots_roi[iSegment]
        betas, cost, nEvals = optimise_first_segment(segment,input_segment,output_segment,support_segment)
        totalEvaluations.append(nEvals)
        all_betas.append(betas)
        all_costs.append(cost)
        toc = tm.time()
        print('Iteration ' + str(iSegment) + ' is finished after '+ str(nEvals) +' evaluations. Elapsed time: ' + str(toc-tic) + 's.')
        # check collocation solution against truth
        model_output_fit_at_truth = model_bsplines.simulate(betas,knots_roi[iSegment], times_roi[iSegment])
        state_at_truth, state_deriv_at_truth, rhs_truth = np.split(model_output_fit_at_truth, 3, axis=1)
        current_model_at_truth = g * state_at_truth[:, 0] * state_at_truth[:, 1] * (voltage_roi[iSegment] - EK)
        fig, axes = plt.subplot_mosaic([['a)', 'a)'], ['b)', 'c)'], ['d)', 'e)']], layout='constrained')
        y_labels = ['I', '$\dot{a}$', '$\dot{r}$', '$a$', '$r$']
        axes['a)'].plot(times_roi[iSegment], output_segment, '-k', label=r'Current true', linewidth=2, alpha=0.7)
        axes['a)'].plot(times_roi[iSegment], current_model_at_truth, '--b', label=r'Optimised given true $\theta$')
        axes['b)'].plot(times_roi[iSegment], rhs_truth[:, 0], '-m', label=r'$\dot{a}$ given true $\theta$', linewidth=2,
                        alpha=0.7)
        axes['b)'].plot(times_roi[iSegment], state_deriv_at_truth[:, 0], '--b',
                        label=r'B-spline derivative given true $\theta$')
        axes['c)'].plot(times_roi[iSegment], rhs_truth[:, 1], '-m', label=r'$\dot{r}$ given true $\theta$', linewidth=2,
                        alpha=0.7)
        axes['c)'].plot(times_roi[iSegment], state_deriv_at_truth[:, 1], '--b',
                        label=r'B-spline derivative given true $\theta$')
        axes['d)'].plot(times_roi[iSegment], states_roi[iSegment][0, :], '-k', label=r'$a$ true', linewidth=2, alpha=0.7)
        axes['d)'].plot(times_roi[iSegment], state_at_truth[:, 0], '--b', label=r'B-spline approximation given true $\theta$')
        axes['e)'].plot(times_roi[iSegment], states_roi[iSegment][1, :], '-k', label=r'$r$ true', linewidth=2, alpha=0.7)
        axes['e)'].plot(times_roi[iSegment], state_at_truth[:, 1], '--b', label=r'B-spline approximation given true $\theta$')
        iAx = 0
        for _, ax in axes.items():
            ax.set_ylabel(y_labels[iAx], fontsize=12)
            ax.legend(fontsize=12, loc='upper left')
            iAx += 1
        # plt.tight_layout(pad=0.3)
        plt.savefig('Figures/cost_terms_at_truth_segment_'+ str(iSegment) + '.png', dpi=400)
        # save the final value of the segment
        end_of_roi.append(state_at_truth[-1,:])
        for iState, stateName in enumerate(state_names):
            state_of_roi[stateName] += list(state_at_truth[:, iState])
    ####################################################################################################################
    #  optimise the following segments by matching the first B-spline height to the previous segment
    for iSegment in range(1,4):
        tic = tm.time()
        segment = times_roi[iSegment]
        input_segment = voltage_roi[iSegment]
        output_segment = current_roi[iSegment]
        support_segment = knots_roi[iSegment]
        collocation_segment = collocation_roi[iSegment]
        # find the scaling coeff of the first height by matiching its height at t0 of the segment to the final value of the previous segment
        first_spline_coeff_a = end_of_roi[-1][0] / collocation_segment[0, 0]
        first_spline_coeff_r = end_of_roi[-1][1] / collocation_segment[0, 0]
        betas, cost, nEvals = optimise_segment(segment,input_segment,output_segment,support_segment)
        totalEvaluations.append(nEvals)
        all_betas.append(betas)
        all_costs.append(cost)
        toc = tm.time()
        print('Iteration ' + str(iSegment) + ' is finished after '+ str(nEvals) +' evaluations. Elapsed time: ' + str(toc-tic) + 's.')
        # check collocation solution against truth
        model_bsplines = bsplineOutputTest()
        model_output_fit_at_truth = model_bsplines.simulate(betas,knots_roi[iSegment], times_roi[iSegment])
        state_at_truth, state_deriv_at_truth, rhs_truth = np.split(model_output_fit_at_truth, 3, axis=1)
        current_model_at_truth = g * state_at_truth[:, 0] * state_at_truth[:, 1] * (voltage_roi[iSegment] - EK)
        fig, axes = plt.subplot_mosaic([['a)', 'a)'], ['b)', 'c)'], ['d)', 'e)']], layout='constrained')
        y_labels = ['I', '$\dot{a}$', '$\dot{r}$', '$a$', '$r$']
        axes['a)'].plot(times_roi[iSegment], output_segment, '-k', label=r'Current true', linewidth=2, alpha=0.7)
        axes['a)'].plot(times_roi[iSegment], current_model_at_truth, '--b', label=r'Optimised given true $\theta$')
        axes['b)'].plot(times_roi[iSegment], rhs_truth[:, 0], '-m', label=r'$\dot{a}$ given true $\theta$', linewidth=2,
                        alpha=0.7)
        axes['b)'].plot(times_roi[iSegment], state_deriv_at_truth[:, 0], '--b',
                        label=r'B-spline derivative given true $\theta$')
        axes['c)'].plot(times_roi[iSegment], rhs_truth[:, 1], '-m', label=r'$\dot{r}$ given true $\theta$', linewidth=2,
                        alpha=0.7)
        axes['c)'].plot(times_roi[iSegment], state_deriv_at_truth[:, 1], '--b',
                        label=r'B-spline derivative given true $\theta$')
        axes['d)'].plot(times_roi[iSegment], states_roi[iSegment][0, :], '-k', label=r'$a$ true', linewidth=2, alpha=0.7)
        axes['d)'].plot(times_roi[iSegment], state_at_truth[:, 0], '--b', label=r'B-spline approximation given true $\theta$')
        axes['e)'].plot(times_roi[iSegment], states_roi[iSegment][1, :], '-k', label=r'$r$ true', linewidth=2, alpha=0.7)
        axes['e)'].plot(times_roi[iSegment], state_at_truth[:, 1], '--b', label=r'B-spline approximation given true $\theta$')
        iAx = 0
        for _, ax in axes.items():
            ax.set_ylabel(y_labels[iAx], fontsize=12)
            ax.legend(fontsize=12, loc='upper left')
            iAx += 1
        # plt.tight_layout(pad=0.3)
        plt.savefig('Figures/cost_terms_at_truth_segment_'+ str(iSegment) + '.png', dpi=400)
        # store end of segment and the whole state for the
        end_of_roi.append(state_at_truth[-1, :])
        for iState, stateName in enumerate(state_names):
            state_of_roi[stateName] += list(state_at_truth[:, iState])
    #### end of loop
    ################################################################################################################
    big_toc = tm.time()
    print('Total evaluations: ' + str(sum(totalEvaluations)) + '. Total runtime: ' + str(big_toc-big_tic) + ' s.' )


    times_of_segments = np.hstack(times_roi[:iSegment+1])
    states_of_segments = [v for k, v in state_of_roi.items()]
    fig, axes = plt.subplot_mosaic([['a)'], ['b)'], ['c)']], layout='constrained')
    y_labels = ['I', '$a$', '$r$']

    axes['a)'].plot(times, observation(times, solution.sol(times), thetas_true), '-k', label=r'Current true', linewidth=2, alpha=0.7)
    axes['a)'].plot(times_of_segments, observation(times_of_segments, np.array(states_of_segments), thetas_true), '--b', label=r'Optimised given true $\theta$')
    axes['a)'].set_xlim(times_of_segments[0],times_of_segments[-1])
    axes['b)'].plot(times_of_segments, solution.sol(times_of_segments)[0, :], '-k', label=r'$a$ true', linewidth=2, alpha=0.7)
    axes['b)'].plot(times_of_segments, state_of_roi[state_names[0]], '--b', label=r'B-spline approximation given true $\theta$')
    axes['c)'].plot(times_of_segments, solution.sol(times_of_segments)[1, :], '-k', label=r'$r$ true', linewidth=2, alpha=0.7)
    axes['c)'].plot(times_of_segments, state_of_roi[state_names[1]], '--b', label=r'B-spline approximation given true $\theta$')
    iAx = 0
    for _, ax in axes.items():
        ax.set_ylabel(y_labels[iAx], fontsize=12)
        ax.legend(fontsize=12, loc='upper left')
        iAx += 1
    # plt.tight_layout(pad=0.3)
    plt.savefig('Figures/states_all_segments.png', dpi=400)
    print('pause here')
