from setup import *
import pints

class Boundaries(pints.Boundaries):
    """
    Boundary constraints on the parameters.

    Arguments:

    ``lower_conductance``
        The lower bound on conductance to use. The upper bound will be set as
        ten times the lower bound.
        Set to ``None`` to use an 8-parameter boundary.
    ``search_transformation``
        A transformation on the parameter space.
        Calls to :meth:`check(p)` will assume ``p`` is in the transformed
        space. Similarly, :meth:`sample()` will return samples in the
        transformed space (although the type of sampling will depend on the
        ``sample_transformation``.
    ``sample_transformation``
        A transformation object, specifying the space to sample in.

    """
    def __init__(
            self, search_transformation, sample_transformation,
            lower_conductance=None):

        super(Boundaries, self).__init__()

        # Include conductance parameter
        self._conductance = (lower_conductance is not None)

        # Parameter transformations
        self._search_transformation = search_transformation
        self._sample_transformation_code = sample_transformation.code()

        # Conductance limits
        if self._conductance:
            self.lower_conductance = lower_conductance
            self.upper_conductance = 10 * lower_conductance

        # Limits on p1-p8
        self.lower_alpha = 1e-7             # Kylie: 1e-7
        self.upper_alpha = 1e3              # Kylie: 1e3
        self.lower_beta = 1e-7              # Kylie: 1e-7
        self.upper_beta = 0.4               # Kylie: 0.4

        # Lower and upper bounds for all parameters
        self.lower = [
            self.lower_alpha,
            self.lower_beta,
            self.lower_alpha,
            self.lower_beta,
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
            self.upper_alpha,
            self.upper_beta,
            self.upper_alpha,
            self.upper_beta,
        ]

        if self._conductance:
            self.lower.append(self.lower_conductance)
            self.upper.append(self.upper_conductance)

        self.lower = np.array(self.lower)
        self.upper = np.array(self.upper)

        # Limits on maximum reaction rates
        self.rmin = 1.67e-5
        self.rmax = 1000

        # Voltages used to calculate maximum rates
        self.vmin = -120
        self.vmax = 60

    def n_parameters(self):
        return 9 if self._conductance else 8

    def check(self, transformed_parameters):

        debug = False

        # check if parameters are sampled in log space
        if InLogScale:
            # Transform parameters back to decimal space
            parameters =np.exp(transformed_parameters)
        else:
            # leave as is
            parameters = transformed_parameters

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
        p1, p2, p3, p4, p5, p6, p7, p8 = parameters[:8]

        # Check positive signed rates
        r = p1 * np.exp(p2 * self.vmax)
        if r < self.rmin or r > self.rmax:
            if debug:
                print('r1')
            return False
        r = p5 * np.exp(p6 * self.vmax)
        if r < self.rmin or r > self.rmax:
            if debug:
                print('r2')
            return False

        # Check negative signed rates
        r = p3 * np.exp(-p4 * self.vmin)
        if r < self.rmin or r > self.rmax:
            if debug:
                print('r3')
            return False
        r = p7 * np.exp(-p8 * self.vmin)
        if r < self.rmin or r > self.rmax:
            if debug:
                print('r4')
            return False

        return True

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
        return 9 if self._conductance else 8

    def check(self, transformed_parameters):

        debug = False

        # check if parameters are sampled in log space
        if InLogScale:
            # Transform parameters back to decimal space
            parameters = np.exp(transformed_parameters)
        else:
            # leave as is
            parameters = transformed_parameters

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