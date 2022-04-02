# state-estimation

A collection of common state estimation algorithms.

*State estimation* is the process of estimating the internal state of a system, from (noisy/imperfect) measurements of the inputs and outputs.

### Kalman Filter on random stable system

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/andrelimzs/state-estimation/main?labpath=examples%2Fkf-demo.ipynb)

System dynamics are
$$
\dot x = A x + Bu
$$


for a negative semidefinite $A$ (such that the system is stable)

![overview](https://github.com/andrelimzs/state-estimation/blob/main/doc/plots/noisy_estimate.png?raw=true)

### Extended Kalman Filter on Nonlinear Drone

```
...
```



## Kalman Filter

### Notation

- State, $x$
- State transition model, $F$
- Observation model, $H$
- (Optional) Input model, $B$
- Process covariance, $Q$
- Observation covariance, $R$
- Kalman Gain, $K_k$
- A posteriori state estimate, $\hat x[k]$
- A posteriori covariance estimate, $P[k]$

$[k | k-1]$ notation means estimate at time $k$ given measurements at time $k-1$

### System Model

For the system
$$
\begin{aligned}
x[k+1] &= F\ x[k] + w \\
y[k] &= H\ x[k] + v
\end{aligned}
$$

### Predict

Predict the future state
$$
\hat x[k|k-1] = F \hat x[k-1] + Bu[k]
$$

And future covariance 
$$
P[k|k-1] = F\ P[k-1]\ F^T + Q
$$

### Update

Calculate Kalman Gain
$$
K_k = P[k|k-1]\ H^T (H\ P[k|k-1]\ H^T + R)^{-1}
$$
Update estimate using feedback (via the Kalman gain and measurement $z$)
$$
\hat x[k] = \hat x[k|k-1] + K_k (z[k] - H \hat x'[k])
$$
And finally update covariance
$$
P[k|k] = (I - K_k H) P[k|k-1]
$$

And repeat



## Extended Kalman Filter

Exactly the same as the Kalman Filter, except for one additional step. Linearize the nonlinear system and observation dynamics at every time step.

For the system
$$
\begin{aligned}
x[k+1] &= f(x[k],\ u[k]) + w \\
y[k] &= h(x[k]) + v
\end{aligned}
$$

### Calculate Jacobian

Calculate the Jacobian $J$ either

1. Analytically

$$
J = \begin{bmatrix}
\frac{\partial f_1}{\partial x_1} & \dots & \frac{\partial f_1}{\partial x_n} \\
\vdots & \ddots & \vdots \\
\frac{\partial f_m}{\partial x_1} & \dots & \frac{\partial f_m}{\partial x_n} \\
\end{bmatrix}
$$

2. Automatic differentiation (AD)

Autodiff algorithms use the chain rule on elementary operations to calculate the final jacobian. 

This project uses autodiff (https://github.com/HIPS/autograd) to calculate the jacobian for any arbitrary state transition and observation model written as a numpy function.



### Predict and Update

Use the linearized $F$ and $H$ in the standard Kalman filter algorithm.



## Linear State Observer

A state observer estimates the internal state of a system from a model of the system $(A,B,C)$, input $u$ and output $y$. It uses feedback to reconcile difference between the naive model prediction and output measurements.

For the system
$$
\dot{x} = Ax + Bu \\
y = Cx + Du
$$


Construct an estimate $\hat x$ with dynamics
$$
\dot{\hat x} = A \hat{x} + Bu
$$


Add feedback
$$
\dot{\hat x} = A \hat{x} + Bu +L(y - C\hat{x})
$$


Which gives the error equation
$$
\dot{\hat{x}}_{err} = (A-LC)\ \hat x_{err}
$$


Where $L$ can be designed using standard design methods such as pole placement or LQR.

In discrete-time the equation becomes
$$
\hat{x}[k+1] = (A_d - LC)\ \hat{x}[k] + B_d\ u + Ly
$$

