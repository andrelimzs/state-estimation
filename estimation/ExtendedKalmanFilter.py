import autograd.numpy as np
from autograd import make_jvp, jacobian
import numpy.linalg as LA
import control
import types

class ExtendedKalmanFilter:
    """
    Extended Kalman Filter for nonlinear system
    
    Parameters
    ----------
    F: (Nonlinear) State Transition
    H: (Nonlinear) Observation Model
    Q: Process Covariance
    R: Observation Covariance

    B: Nonlinear Input Model (Optional)
    x0: Initial State (Optional)

    """
    def __init__(self, Ts, n, k, F, H, Q, R, B=None, x0=None):
        self.Ts = Ts
        self.n = n
        self.lenF = n+k
        
        # Check if F and H are nonlinear functions
        if isinstance(F, types.FunctionType):
            self.F_is_nonlinear = True
            self.J_F = [ jacobian(F, argnum=i) for i in range(self.lenF) ]
        else:
            self.F_is_nonlinear = False
            self.F = F

        if isinstance(H, types.FunctionType):
            self.H_is_nonlienar = True
            self.J_H = [ jacobian(H, argnum=i) for i in range(n) ]
        else:
            self.H_is_nonlienar = False
            self.H = H
            
        self.Q = Q
        self.R = R
        
        self.has_input = (B is not None)
        if self.has_input:
            self.B = B
        
        if x0:  self.xhat = x0
        else:   self.xhat = np.zeros(self.n)
            
        # Initialise initial covariance to zero
        self.P = np.zeros((self.n, self.n))
        
    def discretize(self,A,B,C):
        """ Discretize system (A, B, C) to (Ad, Bd, C) """
        sysd = control.sample_system( control.StateSpace(A,B,C,0), self.Ts )
        A = sysd.A
        B = sysd.B
        return A,B,C
    
    def get_jacobian(self, x, u=None):
        """
        Calculate numerical jacobian of state transition F and observation model H
        
        """
        if self.F_is_nonlinear:
            F = []
            for i in range(self.lenF):
                F.append( self.J_F[i](*x,*u) )
            jacobian = np.stack(F, axis=1)
            F = jacobian[:,:self.n]
            B = jacobian[:,self.n:]
        else:
            F = self.F
            
        if self.H_is_nonlienar:
            H = []
            for i in range(self.n):
                H.append( self.J_H[i](*x) )
            H = np.stack(H, axis=1)
        else:
            H = self.H
            
        F,B,H = self.discretize(F,B,H)
        return F,B,H
    
    def __call__(self, measurement, u=None):
        """
        Predict and update kalman filter
        
        """
        # Compute Jacobian for F and H
        F,B,H = self.get_jacobian(self.xhat, u)
        
        # Predict
        if self.has_input:
            self.xhat = F @ self.xhat + B @ u
        else:
            self.xhat = F @ self.xhat
            
        self.P = F @ self.P @ F.T + self.Q
        
        # Calculate Kalman Gain
        # print(H.shape, self.P.shape, self.R.shape)
        innovation = H @ self.P @ H.T + self.R
        K = self.P @ H.T @ LA.pinv(innovation)
        
        # Update estimate
        self.xhat = self.xhat + K @ (measurement - H @ self.xhat)
        
        # Update covariance
        self.P = (np.eye(self.n) - K @ H) @ self.P
        
        return self.xhat

    # @classmethod
    # def from_continuous(cls, A, C, Q, R, Ts, method='zoh', B=None, x0=None):
    #     """
    #     Initialise Kalman Filter from continuous system description (F,B,H)
    #     by discretization
        
    #     """
    #     N = A.shape[0]
    #     sys = control.StateSpace(A,B,C, np.zeros((N,1)) )

    #     sysd = control.sample_system(sys, Ts, method)
    #     A = sysd.A
    #     B = sysd.B

    #     kf = cls(A,C,Q,R,B=B,x0=x0)

    #     return kf