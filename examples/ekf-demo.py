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

# import numpy as np
# import numpy.linalg as LA
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import control

import autograd.numpy as np
import autograd.numpy.linalg as LA
from autograd import grad


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
    alpha = u[0]
    T = [1]
    gravity = 0
    
    dydt = np.zeros(6)
    dydt[0] = vx + gravity
    dydt[1] = vy
    dydt[2] = omega
    dydt[3] = np.cos(theta + alpha + 0.1*np.random.rand()) * T + 0.2*np.random.rand()
    dydt[4] = np.sin(theta + alpha + 0.1*np.random.rand()) * T + 0.2*np.random.rand()
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
    alpha = ( np.sin(t) - np.sin(3*t) ).reshape((-1))
    ref = np.stack([ alpha, np.ones_like(alpha) ])
    return ref


# +
N = 6
Ts = 1/400

t, y_true, u_true = simulate_system([0,10], ref_sine, Ts)

# Add noise
y = y_true + np.random.randn(*y_true.shape) * np.array([0.5, 0.5, 0.1, 0.05, 0.05, 0.01]).reshape(-1,1)
u = u_true + 0 * np.random.randn(*u_true.shape)


# + tags=[]
def plot_system(t, y, y_true, estimate1=None, estimate2=None, dpi=None):
    fig, axes = plt.subplots(2,3, figsize=(16,6), dpi=dpi)
    ax = axes.flatten()

    labels = ['x', 'y', r'$\theta$', 'vx', 'vy', r'$\omega$']
    units  = ['m', 'm', 'rad', 'm/s', 'm/s', 'rad/s']
    for i,a in enumerate(ax):
        a.plot(t, y[i,:])
        a.plot(t, y_true[i,:], 'k--')
        if estimate1 is not None:
            a.plot(t, estimate1[i,:])
        if estimate2 is not None:
            a.plot(t, estimate2[i,:])

        a.set_ylabel(f'{labels[i]} ({units[i]})')
        # a.set_title(labels[i])
        a.grid()

    axes[0,0].legend(['measurement', 'true', 'estimate'], loc='upper left')
    [ axes[1,i].set_xlabel('time (s)') for i in range(3) ]

    fig.tight_layout()

def plot_control_input(u, u_true, dpi=None):
    fig, axes = plt.subplots(1, 2, figsize=(12,3), dpi=dpi)
    ax = axes.T.flatten()

    for i,a in enumerate(ax):
        a.plot(t, u[i,:])
        a.plot(t, u_true[i,:], 'k--')
        a.grid()

    axes[0].set_ylabel(r'$\alpha$ (rad)')
    axes[1].set_ylabel('T (norm)')
    axes[0].set_xlabel('time (s)')
    axes[1].set_xlabel('time (s)')
    
plot_system(t, y, y_true, dpi=300)
plot_control_input(u, u_true)
# -

# ## Initialise and Use Kalman Filter

# +
# %%time
from estimation.KalmanFilter import KalmanFilter
from estimation.ExtendedKalmanFilter import ExtendedKalmanFilter

# # Instantiate Kalman Filter
A = np.array([[ 0,0,0, 1,0,0 ],
              [ 0,0,0, 0,1,0 ],
              [ 0,0,0, 0,0,1 ],
              [ 0,0,0, 0,0,0 ],
              [ 0,0,0, 0,0,0,],
              [ 0,0,0, 0,0,0 ]])
B = np.array([ [0,0], 
               [0,0],
               [0,0],
               [0,1],
               [1,0],
               [-1,0] ])

H = np.diag([ 0,0,0,1,1,1 ])
# Process Covariance
Q = 0.1 * np.diag([ 1,1,1,1,1,1 ])
# Observation Covariance
R = 1e2 * np.diag([ 1,1,1, 1,1,0.1 ])


def F(x,y,theta,vx,vy,omega,alpha,T):
    alpha = 0
    T = 0
    # State is { x,y,theta, vx,vy,omega }
    dydt = []
    dydt.append( vx )
    dydt.append( vy )
    dydt.append( omega )
    dydt.append( np.cos(theta + alpha) * T )
    dydt.append( np.sin(theta + alpha) * T )
    dydt.append(-np.sin(alpha) * T )
    return np.stack(dydt)

kf = KalmanFilter.from_continuous(A,H,Q,R,Ts,B=B)
ekf = ExtendedKalmanFilter(Ts,N,2,F,H,Q,R)

estimate_kf = np.zeros_like(y)
estimate_ekf = np.zeros_like(y)
for i in range(y.shape[1]):
    measurement = y[:,i]
    control_input = u[:,i]
    # print(f"meas: {measurement.shape} \t u: {control_input.shape}")
    estimate_kf[:,i] = kf(measurement, control_input)
    estimate_ekf[:,i] = ekf(measurement, control_input)
    # kf(measurement, control_input)

plot_system(t, y, y_true, estimate_kf, estimate_ekf, dpi=300)
# -


