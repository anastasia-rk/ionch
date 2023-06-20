# imports

from setup import *
import pints

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

def observation(t, x, theta):
    # I
    a, r = x[:2]
    *ps, g = theta[:9]
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

    # tlim = [0, int(volt_times[-1]*1000)]
    tlim = [0, 2000]
    times = np.linspace(*tlim, tlim[-1])
    # get time points to compute the fit to ODE cost
    times_q = times[[(it % 5 == 0) for it in range(len(times))]]
    volts_new = V(times)
    d2v_dt2 = np.diff(volts_new, n=2)
    switchpoints_new = np.abs(d2v_dt2) > 1e-6

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

    # parameter values for the model
    EK = -80
    p_true = [2.26e-4, 0.0699, 3.45e-5, 0.05462, 0.0873, 8.91e-3, 5.15e-3, 0.03158, 0.1524]
    # initialise and solve ODE
    y0 = [0, 1]
    solution = sp.integrate.solve_ivp(ion_channel_model, tlim, y0, args=[p_true], dense_output=True)
    x_ar = solution.sol(times)
    current = observation(times, x_ar, p_true)

    # create B-spline representation
    # build the collocation matrix using the defined knot structure
    degree = 3
    fig, ax = plt.subplots()
    outer = [knots_all[0], knots_all[0], knots_all[0], knots_all[-1], knots_all[-1], knots_all[-1]]
    outer_y = []
    knots = np.insert(outer, 3, knots_all)  # create knots for which we want to sbuild splines
    coeffs = np.zeros(len(knots) - degree - 1)  # number of splines will depend on the knot order
    spl_ones = BSpline(knots, np.ones_like(coeffs), degree)
    tau = np.arange(knots[0], knots[-1] + 1)
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

########################################################################################################################
    # create pints models
    nBsplineCoeffs = len(coeffs) * 2 # this to be used in params method of class ForwardModel
    nOutputs = 2 # this to be used in MultiOutput problem of pints
    nTimes = len(times_q) # this to be used in MultiOutput problem of pints
    class bsplineModel(pints.ForwardModel):
        def simulate(self, parameters, times):
            # Run a simulation with the given parameters for the
            # given times and return the simulated values
            coeffs_a, coeffs_r = np.split(parameters, 2)
            # collocation matrix is the fastest way to compute the whole curve at measurment points
            a = coeffs_a @ collocation
            r = coeffs_r @ collocation
            *ps, g = p_true[:9]
            volts_for_model = V(times)
            y_model = g * a * r * (volts_for_model - EK)
            return y_model
        def n_parameters(self):
            # Return the dimension of the parameter vector
            return nBsplineCoeffs

    class derivModel(pints.ForwardModel):
        def simulate(self, parameters, times):
            # Run a simulation with the given parameters for the
            # given times and return the simulated values
            coeffs_a, coeffs_r = np.split(parameters, 2)
            ## model descrepancy - bear in mind that times now stands for quadrature time points, times_q!
            tck_a = tuple([knots, coeffs_a, degree])
            tck_r = tuple([knots, coeffs_r, degree])
            dot_a = sp.interpolate.splev(times, tck_a, der=1)
            dot_r = sp.interpolate.splev(times, tck_r, der=1)
            fun_a = sp.interpolate.splev(times, tck_a, der=0)
            fun_r = sp.interpolate.splev(times, tck_r, der=0)
            # the derivatives must be put into an array
            dadr = []
            for it, t_q in enumerate(times):
                x = [fun_a[it], fun_r[it]]
                dadr.append(ion_channel_model(t_q, x, p_true))
            rhs_theta = np.array(dadr)
            spline_deriv = np.transpose(np.array([dot_a, dot_r]))
            # output the discrepancy - we will be comparing it to zeros!
            return spline_deriv - rhs_theta
        def n_parameters(self):
            # Return the dimension of the parameter vector
            return nBsplineCoeffs
        def n_outputs(self):
            # Return the dimension of the output vector
            return nOutputs

    ## attempt to define the cost in pints
    # define the model where the output is the current
    model_1 = bsplineModel()
    # create the problem of comparing the modelled current with measured current
    problem_1 = pints.SingleOutputProblem(model=model_1,times=times, values=current)
    # associate the cost with it
    error_1 = pints.SumOfSquaresError(problem=problem_1)
    # create the problme of comparing the modelled derivatives to the ODE rhs
    # here we have a problem because the unknown params enter both terms in the error computation.
    # dot(bSplines*Theta_splines) - rhs(bSplines*Theta_splines,Theta_odes)
    # we reframe this into pints notattion by introducing:
    tricky_zeros = np.zeros([nTimes,nOutputs])
    # that will be compared with the output of
    model_2 = derivModel()
    # which will be the error between the two sides of ODE using the B-spline surface
    # we thus end up with the following error
    # (dot(bSplines*Theta_splines) - rhs(bSplines*Theta_splines,Theta_odes) - tricky_zeros)
    problem_2 = pints.MultiOutputProblem(model=model_2, times=times_q, values=tricky_zeros)
    error_2 = pints.SumOfSquaresError(problem=problem_2)
    # combine the two costs as a weighted sum of errors
    lambd = 10 # give a higher weight to the modelling descripancy to make data error a penalty
    total_error = pints.SumOfErrors(error_measures=(error_1,error_2),weights=(1, lambd))
    # define an initial point for the unknown B-spline coeffs:
    init_spline_betas = 0.01*np.ones(nBsplineCoeffs)
    # create an optimisation controller and run
    optimiser = pints.OptimisationController(total_error,x0=init_spline_betas,method=pints.CMAES)
    optimiser.set_threshold(1)
    optimiser.set_max_iterations(20000)
    optimiser.set_max_unchanged_iterations(5)
    found_parameters, found_value = optimiser.run()
    # evaluate splines and derivatives in with the optimised parameter values:
    coeffs_a, coeffs_r = np.split(found_parameters, 2)
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
    # plot the output
    fig, axes = plt.subplots(2, 2, sharex=True)
    y_labels = ['a', '$\dot{a}$', 'r', '$\dot{r}$']
    axes[0, 0].plot(x_ar[0], '-k', label='true')
    axes[0, 0].plot(fun_a, '--r', label='B-splines')
    axes[1, 0].plot(x_ar[1], '-k', label='true')
    axes[1, 0].plot(fun_r, '--r', label='B-splines')
    axes[0, 1].plot(rhs_theta[:, 0], '-k', label='RHS')
    axes[0, 1].plot(dot_a, '--r', label='B-spline derivative')
    axes[1, 1].plot(rhs_theta[:, 1], '-k', label='RHS')
    axes[1, 1].plot(dot_r, '--r', label='B-spline derivative')
    for iAx, ax in enumerate(axes.flatten()):
        ax.legend(fontsize=14, loc='upper right')
        ax.set_ylabel(y_labels[iAx])
    plt.tight_layout()
    plt.savefig('Figures/LS_b_splin_grid.png')
    print('pause here')