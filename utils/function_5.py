import numpy as np 
from week1 import Pi,PiInv 
from scipy import optimize


def triangulate(qn,pn):
    """ Input 
        Q: list of n pixel coordinates (q1,q2,...qn)
        Pn: list of n projection matrices 
        output: triangulation of the 3d point
    """
    
    if len(qn) != len(pn):
        raise ValueError("Expected lists of equal length, len(Q)!=len(pn)")

    
    B_i = lambda P,q: np.array([  [P[2]*q[0] - P[0]],[P[2]*q[1] - P[1]] ])

    B = np.array((([B_i(P_i,q_i) for P_i, q_i in zip(pn,qn)])))

    B = B.reshape(len(pn)*2,4)

    U,S,VT = np.linalg.svd(B)

    Q = VT[-1,:].T
    return Q


    


def triangulate_nonlinear(qs,ps):

    x0 = triangulate(qs,ps)
    compute_residuals(x0)

import numpy as np 


def triangulate(qn,pn):

    """ Input 
        Q: list of n pixel coordinates (q1,q2,...qn)
        Pn: list of n projection matrices 
        output: triangulation of the 3d point
    """
    
    if len(qn) != len(pn):
        raise ValueError("Expected lists of equal length, len(Q)!=len(pn)")

    
    B_i = lambda P,q: np.array([  [P[2]*q[0] - P[0]],[P[2]*q[1] - P[1]] ])

    B = np.array((([B_i(P_i,q_i) for P_i, q_i in zip(pn,qn)])))

    B = B.reshape(len(pn)*2,4)

    U,S,VT = np.linalg.svd(B)

    Q = VT[-1,:].T
    return Q


    


def triangulate_nonlinear(qs,ps):

    def compute_residuals(Q):

        row = lambda P,q: Pi(P@PiInv(Q))- q.T
        
        f_Q = np.hstack([row(P,q) for P,q in zip(ps,qs)])
        return f_Q.T.flatten()



    x0 = Pi(triangulate(qs,ps))
    x = optimize.least_squares(compute_residuals,x0)
    return x 
