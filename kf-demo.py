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

import numpy as np
import numpy.linalg as LA
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import control


# ## Simulate random stable system

# +
def simulate_system(A, B, t_span, ref, Ts, y0=None):
    N = A.shape[0]
    
    if y0 is None:
        y0 = np.zeros(N)
        
    # Form linear system x_dot = Ax + Bu
    def eqn(t,y,ref):
        u = ref(t)
        dydt = A@y + B@u
        return dydt
    
    # Set ODE parameters
    t_eval = np.arange(t_span[0], t_span[1], Ts)
    
    # Simulate system by solving ODE
    sol = solve_ivp(eqn, t_span, y0, t_eval=t_eval, args=(ref,))
    
    return sol.t, sol.y, ref(sol.t)

def ref_sine(t):
    return ( np.sin(t) + np.sin(3*t) ).reshape((-1))


# +
Ts = 0.01
N = 5

# Generate random stable system
A = np.random.rand(N,N)
A = -A @ A.T
B = np.random.rand(N,1)

t, y_true, u_true = simulate_system(A, B, [0,20], ref_sine, Ts)

# Add noise
y = y_true + np.random.randn(*y_true.shape) * (0.05 * np.random.rand(N,1) + 0.05)
u = u_true + 0.1 * np.random.randn(*u_true.shape)


# fig, ax = plt.subplots(N+1, figsize=(12,9))
# fig.tight_layout()
# for i in range(N):
#     ax[i].plot(y[i,:])
#     ax[i].grid()

# ax[N].plot(u)
# ax[N].grid()
# -

# ## Initialise and Use Kalman Filter

# +
# %%time

from KalmanFilter import KalmanFilter

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

fig,ax = plt.subplots(N, figsize=(12,9))
fig.tight_layout()
for i in range(N):
    ax[i].plot(t, y[i,:])
    ax[i].plot(t, estimate[i,:])
    ax[i].plot(t, y_true[i,:], '--')
    ax[i].grid()
# +
# import cvxpy as cp

# N = y.shape[1]

# X = cp.Variable((2,N))

# # Constraints
# constraint = [ ]

# # Objective
# obj = cp.Minimize(
# cp.sum(
#     cp.norm(
#         A@X + B@u.reshape((1,-1)) - y
#     )
# ))

# # Problem
# prob = cp.Problem(obj, constraint)

# prob.solve(solver='MOSEK')

# print("status:", prob.status)
# print("optimal value", prob.value)
# -

