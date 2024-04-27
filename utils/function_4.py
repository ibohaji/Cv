import numpy as np 
from week1 import * 


def PiInv(coordinate):
    """
    Convert inhomogeneous coordinates to homogeneous coordinates.

    Parameters:
    - coordinate: A numpy array of inhomogeneous coordinates. This can be a 1D array (vector) or a 2D array (matrix).

    Returns:
    - A numpy array of homogeneous coordinates. For a vector, appends a 1 to the end. For a matrix, appends a row of 1s.
    """

    # Handle 1D array (vector) case
    if coordinate.ndim == 1:
        hom = np.append(coordinate, 1)
    # Handle 2D array (matrix) case
    elif coordinate.ndim == 2:
        w = np.ones((1, coordinate.shape[1]))
        hom = np.vstack((coordinate, w))
    else:
        raise ValueError("Input must be a 1D (vector) or 2D (matrix) array.")

    return hom

def crossOp(p): 
    p_cross = np.array([[0,-p[2],p[1]],[p[2],0,-p[0]],[-p[1],p[0],0]]) 
    return p_cross

def pest(Q,q):
        """ 
        Estimate the projection matrix using DLT 
        Paremeters: 
        Q: 3D homogenous coordinates 
        q: the projected 2d homogenous coordinate
        returns: Projection matrix P 
        """

        N,M = Q.shape
        B = None

        for Q_i,q_i in zip(Q.T,q.T):
              
              
              if B is not None:
                    B = np.vstack((B,np.kron(Q_i,crossOp(PiInv(q_i)))))

              else:
                    B = np.kron(Q_i,crossOp(PiInv(q_i)))

        U,S,VT = np.linalg.svd(B) 
        P = VT[-1,:].reshape(4,3).T

        return P 


def RMSE(q_hat,q): 
    if q_hat.shape != q.shape:
        raise ValueError("Incompatible shapes, q_hat != q")

    _,n = q_hat.shape
    rmse = np.sqrt(np.sum((q_hat - q)**2/n))
    return rmse

def checkerboard_points(n, m):
    def Q_ij(i, j):
        return np.array([i - (n-1)/2, j - (m-1)/2, 0])
    
    points = np.zeros((3, n*m))
    idx = 0
    for i in range(n):
        for j in range(m):
            points[:, idx] = Q_ij(i, j)
            idx += 1
    
    return points




def smart_transpose(arr):

    """
    Transposes a numpy array.
    - If the array is 1D, it converts it to a 2D column vector.
    - If the array is 2D, it transposes it.
    """

    # If it's a 1D array, reshape it to a 2D column vector
    if arr.ndim == 1:
        return arr.reshape(-1, 1).T
    # If it's already 2D, just transpose it
    elif arr.ndim == 2:
        return arr.T
    # For arrays with more dimensions, you might need a more specific handling
    else:
        raise ValueError("Array with more than 2 dimensions are not supported.")





def hest(q1,q2):
    


   # q1_ix = [crossOp(q_i) for q_i in q1.T]
    B = None 


    for q1_i,q2_i in zip(q1.T,q2.T):

        if B is not None: 
            B = np.vstack((B,np.kron(q2_i, crossOp(q1_i)) ))
        else:
            B = np.kron((q2_i), crossOp(q1_i))

    U,S,VT = np.linalg.svd(B)
   
    H = VT[-1,:].reshape(3,3).T

    return H




def estimateHomograpies(qs,Qs):
    """
    Parameters
    ----------
    Q_omega: an array original un-transformed checkerboard points in 3D, for example QΩ.
    qs: a list of arrays, each element in the list containing QΩ projected to the image plane from different views, for example qs could be [qa, qb, qc].
    Output
    ------
    return the homographies that map from Q_omega to each of the entries in qs. The homographies should work as follows:
    """
    
    Homograhpies = []
    Qs[-1,:] = 1
  

    for _,qi in zip(Qs,qs):

        H_i = hest(PiInv(qi),Qs)
        Homograhpies.append(H_i)

    return Homograhpies



def getVij(hi, hj):
    Vij = np.array([ hi[0]*hj[0], hi[0]*hj[1] + hi[1]*hj[0], hi[1]*hj[1], hi[2]*hj[0] + hi[0]*hj[2], hi[2]*hj[1] + hi[1]*hj[2], hi[2]*hj[2] ])
   
   
    return Vij.T


def getV(Hs):
    v = []
    for H in Hs:
        h1 = H[:,0]
        h2 = H[:,1]

        v12 = getVij(h1, h2)
        v11 = getVij(h1, h1)
        v22 = getVij(h2, h2)
        v.append(v12.T)
        v.append((v11 - v22).T)

    return np.array(v)

def getB(b):
    B = np.zeros((3,3))
    B[0,0] = b[0]
    B[0,1] = b[1]
    B[0,2] = b[3]
    B[1,0] = b[1]
    B[1,1] = b[2]
    B[1,2] = b[4]
    B[2,0] = b[3]
    B[2,1] = b[4]
    B[2,2] = b[5]

    return B 



def estimate_b(Hs):
    """ 
    Parameters
    ----------
    Hs: List of 3x3 Homographies, Hs = [H1,H2,...Hn] 

    Returns 
    -------
    1x6 vector b 
    """
    V = getV(Hs)
    U,S,VT = np.linalg.svd(V)
    B = VT[-1,:]

    return B 


def get_hij(hi,hj):

    vij = [hi[0]*hj[0],
           hi[0]*hj[1] + hi[1]*hj[0],
           hi[1]*hj[1],
           hi[2]*hj[0] + hi[0]*hj[2],
           hi[2]*hj[1] + hi[1]*hj[2],
           hi[2]*hj[2] ]
    
    return (np.array(vij).reshape(1,-1))


def estimate_b_diagnostics(Hs):
    """ 
    Parameters
    ----------
    Hs: List of 3x3 Homographies, Hs = [H1,H2,...Hn] 

    Returns 
    -------
    1x6 vector b 
    """

    Vs= []
   
    V = []


    for H in Hs:
            h1,h2,h3 = H[:,0],H[:,1],H[:,2]
            v12 = get_hij(h1,h2)
            v11 = get_hij(h1,h1) 
            v22 = get_hij(h2,h2)
            print("ze shape v12:{}".format(v12.shape))
            V.append(v12)
            V.append((v11-v22))

    print("changed")
    print("shapezzz V: {}".format(np.array(V).shape))
    V = np.array(V)  
    U,S,VT = np.linalg.svd(V)
    B = VT[-1,:]

    return B 




def get_B(b):
    s,v_0,alpha,beta,gama,u_0 = extract_parameters(b)
    B = np.array([[1/alpha**2,-gama / (alpha**2 * beta),
     (v_0*gama-u_0*beta / (alpha**2 * beta))],
     [-gama / (alpha**2 * beta),gama**2 / (alpha**2 * beta**2) + 1 / beta**2,
      -gama*(v_0*gama-u_0 * beta)/(alpha**2 * beta **2) - v_0/beta**2],
      [(v_0 * gama - u_0*beta) / (alpha** 2 * beta),-gama*((v_0*gama - u_0 * beta) / (alpha ** 2 * beta**2)-v_0/beta**2),
      ((v_0*gama - u_0*beta)**2)/(alpha**2 * beta**2) + v_0**2/beta**2 + 1 ]])
    return B


# 0    1   2   3   4   5
#[B11,B12,B22,B13,B23,B33].T
def extract_parameters(B):
     
     B11,B12,B22,B13,B23,B33 = B

     v_0 = ( B[1]*B[3] - B[0] * B[4] ) / ( B[0]*B[2]-B[1]**2 )

     scale_factor = B33 - (B13**2 + v_0*(B12*B13 - B11*B23))/B11  #B[5] - ( B[3]**2 + v_0*( B[1] * B[3]-B[0] * B[4] ) / B[0] ) 
     alpha = np.sqrt( scale_factor/B[0] )
    
     beta = np.sqrt( scale_factor * B[0] / ( B[0] * B[2] - B[1]**2 ) )
     
     gama = -B[1] * alpha**2 * beta/scale_factor
     u_0 = gama*v_0/beta - B[3]*alpha**2/scale_factor


     return scale_factor,v_0,alpha,beta,gama,u_0


def get_CameraM(b): 
     scale_factor,v_0,alpha,beta,gama,u_0 = extract_parameters(b) 
     print("lambda:{}\nv_0:{}\nalpha:{}\nbeta:{}\ngama:{}\nu_0:{}".format(scale_factor,v_0,alpha,beta,gama,u_0))
     A = np.array([[alpha,gama,u_0],[0,beta,v_0],[0,0,1]])
     return A 


def estimateIntrisics(Hs):
    b = estimate_b(Hs)
    K = get_CameraM(b)
    return K 


def estimateExtrinsics(K,Hs):
    
    translations = []
    rotations= [] 
    n = 1 
    for H in Hs:
    

        lambda_ = 1 / np.linalg.norm(np.linalg.inv(K) @ H[:,0])
   
        r1 = lambda_ * np.linalg.inv(K) @ H[:,0]
        r2 = lambda_ * np.linalg.inv(K) @ H[:,1]
        r3 = np.cross(r1,r2)
        R = np.array([r1,r2,r3])

        t = lambda_ * np.linalg.inv(K) @ H[:,2] 
        R = np.array([r1,r2,r3])

        translations.append(t)
        rotations.append(R)

    return rotations,translations



def calibrateCamera(qs,Q):

    Hs = estimateHomograpies(qs,Q)
    K = estimateIntrisics(Hs)
    R,t = estimateExtrinsics(K,Hs)
    
    return K,R,t



