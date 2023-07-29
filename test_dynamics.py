# imports
import numpy as np
import scipy as sp
from scipy.interpolate import BSpline
from autograd import hessian, jacobian, grad
import matplotlib
from matplotlib import pyplot as plt
import time as tm
plt.rcParams['figure.figsize'] = (20,10)
plt.rcParams['figure.dpi'] = 400
plt.rcParams['axes.facecolor']='white'
plt.rcParams['savefig.facecolor']='white'
plt.style.use("ggplot")
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica"]})


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
    return volts_intepolated((t + 900)/ 1000)

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
    tlim = [0,4000]
    times = np.linspace(*tlim, tlim[-1])
    # get time points to compute the fit to ODE cost
    times_q = times[[(it % 5 == 0) for it in range(len(times))]]
    volts_new = V(times)
    d2v_dt2 = np.diff(volts_new, n=2)
    switchpoints_new = np.abs(d2v_dt2) > 1e-6
    ## uncomment this to see the second derivative of the voltage clamp
    fig, ax = plt.subplots()
    ax.plot(times[2:], d2v_dt2, 'b', label='Voltage clamp')
    ax.legend(fontsize=14)
    ax.set_xlabel('times, s')
    ax.set_ylabel('2nd derivative')
    plt.savefig('Figures/2nd_derivative.png')

    # parameter values for the model
    EK = -80
    p_true = [2.26e-4, 0.0699, 3.45e-5, 0.05462, 0.0873, 8.91e-3, 5.15e-3, 0.03158, 0.1524]
    # initialise and solve ODE
    x0 = [0, 1]
    solution = sp.integrate.solve_ivp(ion_channel_model, tlim, x0, args=[p_true], method='LSODA',rtol=1e-8,atol=1e-8,dense_output=True)
    x_ar = solution.sol(times)
    current = observation(times, x_ar, p_true)

    # plot three states and the output
    fig, axes = plt.subplots(2, 2)
    axes[0, 0].plot(times, x_ar[0,:], 'b')
    axes[0, 0].plot(times[2:][switchpoints_new], x_ar[0,2:][switchpoints_new], 'r.')
    axes[0, 0].set_xlabel('times, ms')
    axes[0, 0].set_ylabel('a gating variable')
    axes[0, 0].set_xlim(tlim)
    axes[0, 1].plot(times, x_ar[1,:], 'b')
    axes[0, 1].plot(times[2:][switchpoints_new], x_ar[1,2:][switchpoints_new], 'r.')
    axes[0, 1].set_xlabel('times, ms')
    axes[0, 1].set_ylabel('r gating variable')
    axes[0, 1].set_xlim(tlim)
    axes[1, 0].plot(times, volts_new, 'b')
    axes[1, 0].plot(times[2:][switchpoints_new], volts_new[2:][switchpoints_new], 'r.')
    axes[1, 0].set_xlabel('times, ms')
    axes[1, 0].set_ylabel('voltage, mV')
    axes[1, 0].set_xlim(tlim)
    axes[1, 1].plot(times, current, 'b')
    axes[1, 1].plot(times[2:][switchpoints_new], current[2:][switchpoints_new], 'r.')
    axes[1, 1].set_xlabel('times, ms')
    axes[1, 1].set_ylabel('Current, A')
    axes[1, 1].set_xlim(tlim)
    plt.tight_layout()
    plt.savefig('Figures/model_states_output.png')