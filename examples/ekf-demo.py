# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.13.7
#   kernelspec:
#     display_name: Python [conda env:control-systems] *
#     language: python
#     name: conda-env-control-systems-py
# ---

# +
# %load_ext autoreload
# %autoreload 2

import sys
sys.path.append("..")

import numpy as np
import numpy.linalg as LA
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import control


# -

# ## Simulate Nonlinear System
#
# Simulate a thrust-vectored drone
# <center>
# <img src="../doc/ekf/thrust-vector-diagram.png" alt="Diagram" style="width: 300px" />
# </center>
#
# $
# \left\{ \begin{aligned}
# \dot{x}\  &= \ v_x \\
# \dot{y}\  &= \ v_y \\
# \dot{\theta}\ &= \ \omega \\
# \dot v_x  &= \ \cos{(\theta + \alpha)}\ T \\
# \dot v_y  &= \ \sin{(\theta + \alpha)}\ T \\
# \dot{\omega}\ &= -\sin \alpha\ T
# \end{aligned} \right.
# $
#

# +
# Form linear system x_dot = Ax + Bu
def eqn(t,y,ref):
    """ Thrust vectored drone """
    # Unpack state
    theta = y[2]
    vx = y[3]
    vy = y[4]
    omega = y[5]
    
    # Unpack control input
    u = ref(t)
    alpha = u
    T = 1
    gravity = -1
    
    dydt = np.zeros(6)
    # dx = vx
    dydt[0] = vx + gravity
    # dy = vy
    dydt[1] = vy
    # dtheta = omega
    dydt[2] = omega
    # dvx = cos(theta + alpha) T
    dydt[3] = np.cos(theta + alpha) * T
    # dvy = sin(theta + alpha) T
    dydt[4] = np.sin(theta + alpha) * T
    # domega = -sin (alpha) T
    dydt[5] = -np.sin(alpha) * T
    
    return dydt

def simulate_system(t_span, ref, Ts, y0=None):
    y0 = np.zeros(6)
    
    # Set ODE parameters
    t_eval = np.arange(t_span[0], t_span[1], Ts)
    
    # Simulate system by solving ODE
    sol = solve_ivp(eqn, t_span, y0, t_eval=t_eval, args=(ref,))
    
    return sol.t, sol.y, ref(sol.t)

def ref_sine(t):
    return ( np.sin(t) + np.sin(3*t) ).reshape((-1))


# +
N = 6
Ts = 0.01

t, y_true, u_true = simulate_system([0,10], ref_sine, Ts)

# Add noise
y = y_true + np.random.randn(*y_true.shape) * np.array([0.5, 0.5, 0.1, 0.1, 0.01, 0.01]).reshape(-1,1)
u = u_true + 0.1 * np.random.randn(*u_true.shape)

# +
fig, axes = plt.subplots(N//2, 2, figsize=(12,9), dpi=100)
ax = axes.T.flatten()

labels = ['x', 'y', r'$\theta$', 'vx', 'vy', r'$\omega$']
units  = ['m', 'm', 'deg', 'm/s', 'm/s', 'deg/s']
for i,a in enumerate(ax):
    a.plot(t, y[i,:])
    a.plot(t, y_true[i,:], 'k--')
    # a.plot(t, estimate[i,:])
    
    a.set_ylabel(f'{labels[i]} ({units[i]})')
    a.set_title(labels[i])
    a.grid()
    
ax[0].legend(['measurement', 'true', 'estimate'], loc='upper left')
ax[2].set_xlabel('time (s)')
ax[5].set_xlabel('time (s)')

fig.tight_layout()
# -

# ## Initialise and Use Kalman Filter

# +
# %%time
from estimation.KalmanFilter import KalmanFilter

# # Instantiate Kalman Filter
H = np.eye(N)
# Process Covariance
Q = 1e-2 * np.eye(N)
# Observation Covariance
R = 1e2 * np.eye(N)

kf = KalmanFilter.from_continuous(A,H,Q,R,Ts,B=B)

estimate = np.zeros_like(y)
for i in range(y.shape[1]):
    measurement = y[:,i]
    control_input = u[i:i+1]
    # print(f"meas: {measurement.shape} \t u: {control_input.shape}")
    estimate[:,i] = kf(measurement, control_input)
    # kf(measurement, control_input)

# Unpack
est_pos = estimate[0,:].T
est_vel = estimate[1,:].T

fig,axes = plt.subplots(2, N//2, figsize=(12,6), dpi=100)
fig.tight_layout()
for i,ax in enumerate(axes.flatten()):
    ax.plot(t, y[i,:])
    ax.plot(t, y_true[i,:], 'k--')
    ax.plot(t, estimate[i,:])
    ax.grid()
    
axes[0,2].legend(['measurement', 'true', 'estimate'], loc='upper right')
_ = [ axes[1,i].set_xlabel('time (s)') for i in range(3) ]
