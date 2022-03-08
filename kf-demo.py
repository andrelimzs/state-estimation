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


# +
def simulate_system(A, B, t_span, ref, Ts, y0=np.zeros(2)):
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

Ts = 0.01
A = np.array([[0, 1],[-0.2, -0.4]])
B = np.array([[-0.2],[1]])

t, y_true, u_true = simulate_system(A, B, [0,20], ref_sine, Ts)

# Add noise
y = y_true + np.array([[0.5], [0.2]]) * np.random.randn(*y_true.shape)
u = u_true + 0.1 * np.random.randn(*u_true.shape)

# Unpack
pos = y[0,:].T
vel = y[1,:].T
# -

_,ax = plt.subplots(3)
ax[0].plot(pos)
ax[1].plot(vel)
ax[2].plot(u)

# +
import numpy.linalg as LA

class KalmanFilter:
    def __init__(self, state_transition, observation_model, process_cov, observation_cov, input_model=None, x0=None):
        # Check matrix dimensions
        if state_transition.shape[0] != state_transition.shape[1]:
            raise ValueError(f"State transition matrix must be square, but received matrix of size {state_transition.shape}")
        
        self.F = state_transition
        self.H = observation_model
        self.Q = process_cov
        self.R = observation_cov
        self.n = self.F.shape[0]
        
        self.has_input = input_model is not None
        if self.has_input:
            self.B = input_model
        
        if x0:
            self.xhat = x0
        else:
            self.xhat = np.zeros(self.n)
            
        # Initialise initial covariance to zero
        self.P = np.zeros((self.n, self.n))
        
    
    def __call__(self, measurement, u=None):
        # Predict
        if self.has_input:
            self.xhat = self.F @ self.xhat + self.B @ u
        else:
            self.xhat = self.F @ self.xhat
            
        self.P = self.F @ self.P @ self.F.T + self.Q
        
        # Calculate Kalman Gain
        innovation = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ LA.pinv(innovation)
        
        # Update estimate
        self.xhat = self.xhat + K @ (measurement - self.H @ self.xhat)
        
        # Update covariance
        self.P = (np.eye(self.n) - K @ self.H) @ self.P
        
        return self.xhat

# +
# Instantiate Kalman Filter
F = np.eye(2) + Ts * A
H = np.eye(2)
# Process Covariance
Q = np.diag([0.1, 0.2]) * 0
# Observation Covariance
R = np.diag([0.1, 0.2]) * 0
Bd = Ts * B
kf = KalmanFilter(F,H,Q,R,Bd)

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

_,ax = plt.subplots(2, figsize=(24,16))
ax[0].plot(y[0,:])
ax[0].plot(estimate[0,:])
ax[0].plot(y_true[0,:], '--')
ax[0].grid()

ax[1].plot(y[1,:])
ax[1].plot(estimate[1,:])
ax[1].plot(y_true[1,:], '--')
ax[1].grid()
# -


