import numpy as np
import numpy.linalg as LA
import control

class KalmanFilter:
    """
    Kalman Filter for a linear system
    
    Parameters
    ----------
    F: State Transition (A matrix)
    H: Observation Model (B matrix)
    Q: Process Covariance
    R: Observation Covariance

    B: Input Model (Optional)
    x0: Initial State (Optional)

    """
    def __init__(self, F, H, Q, R, B=None, x0=None):
        # Check matrix dimensions
        if F.shape[0] != F.shape[1]:
            raise ValueError(f"State transition matrix must be square, but received matrix of size {F.shape}")
        
        self.F = F
        self.H = H
        self.Q = Q
        self.R = R
        self.n = self.F.shape[0]
        
        self.has_input = B is not None
        if self.has_input:
            self.B = B
        
        if x0:
            self.xhat = x0
        else:
            self.xhat = np.zeros(self.n)
            
        # Initialise initial covariance to zero
        self.P = np.zeros((self.n, self.n))
        
    def __call__(self, measurement, u=None):
        """
        Predict and update kalman filter
        
        """
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

    @classmethod
    def from_continuous(cls, A, C, Q, R, Ts, method='zoh', B=None, x0=None):
        """
        Initialise Kalman Filter from continuous system description (F,B,H)
        by discretization
        
        """
        N = A.shape[0]
        sys = control.StateSpace(A,B,C,0)

        sysd = control.sample_system(sys, Ts, method)
        A = sysd.A
        B = sysd.B

        kf = cls(A,C,Q,R,B=B,x0=x0)

        return kf