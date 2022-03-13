# state-estimation

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/andrelimzs/state-estimation/main?labpath=examples%2Fkf-demo.ipynb)

A collection of common state estimation algorithms.

*State estimation* is the process of estimating the internal state of a system, from (noisy/imperfect) measurements of the inputs and outputs.

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

## Kalman Filter

For
$$
x_{k+1} = f(x,y) + w \\
y_k = h(x_k) + v
$$

### 1 Predict

Predict
$$
\hat x_{k+1}' = \Phi \hat x_k + Bu
$$

$$
P_{k+1} = \Phi P_k \Phi^T + Q
$$

### 2 Update

Calculate Kalman Gain
$$
K_k = P_k' H^T (H P_k' H^T + R)^{-1}
$$
Update estimate
$$
\hat x_k = \hat x_k' + K_k (z_k - H \hat x_k')
$$
Update covariance
$$
P_k = (I - K_k H) P_k'
$$

