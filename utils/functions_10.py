import numpy as np 
import cv2 



def Pi(coordinates):
    inhom = coordinates[:-1]/coordinates[-1]
    return inhom 


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
""" 
Implement a RANSAC algorithm for finding the homography between im1 and im2
""" 

# Sample 4 points without replacement  


def crossOp(p): 
    p_cross = np.array([[0,-p[2],p[1]],[p[2],0,-p[0]],[-p[1],p[0],0]]) 
    return p_cross


def distance(H,p1,p2):
    """distapp(H, p1i, p2i) = ‖Π(Hp2i) − Π(p1i)‖2 2 + ‖Π(H−1p1i) − Π(p2i)‖22 ."""
    p1_tilde =  np.linalg.inv(H) @ ( H @ PiInv(p1))
    p2_tilde =  H @ (np.linalg.inv(H)  @ PiInv(p2) )
    
    term1 = Pi(p1_tilde)  - Pi(H @ p2_tilde)
    term2 = Pi(p2_tilde) - Pi(np.linalg.inv(H) @ p1_tilde)
    distance =  np.linalg.norm(term1) + np.linalg.norm(term2)
    return distance
    



def sample_points(points, n):
    """input: set of matches"""
    return np.random.choice(points, n, replace=False)

def hest(q1,q2):
   
    B = None 
    for q1_i,q2_i in zip(q1,q2):
        q1_i = PiInv(q1_i)
        q2_i = PiInv(q2_i)
        
        if B is not None: 
            B = np.vstack((B,np.kron(q2_i, crossOp(q1_i)) ))
        else:
            B = np.kron((q2_i.reshape(1,-1)), crossOp(q1_i))

    U,S,VT = np.linalg.svd(B)
    H = VT[-1,:].reshape(3,3).T
    return H



def estHomographyRANS(des1,des2,kp1,kp2,sigma=3):
    n = 4
    max_iterations = 250
    best_inliers = 0
    best_H = None  
    coordinates = []
    threshold = 5.99 * sigma**2
    
    bf = cv2.BFMatcher_create(crossCheck=True) 
    matches = bf.match(des1, des2) 

    for i in range(max_iterations):
        sample = sample_points(matches, n)
        p1 = np.array([kp1[match.queryIdx].pt for match in sample])
        p2 = np.array([kp2[match.trainIdx].pt for match in sample])
      
        H = hest(p1, p2)
        inliers = 0
        for match in matches:
            p1 = np.array(kp1[match.queryIdx].pt) 
            p2 = np.array(kp2[match.trainIdx].pt)
        
            d = distance(H,p1,p2)
            if d < threshold:

                inliers += 1
        if inliers > best_inliers:
            best_inliers = inliers
            best_H = H
    print("NUmber of best inliers found: {}".format(best_inliers))
    return best_H


def getRange(img1,img2,H):
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]

    corners1 = np.float32([[0, 0], [0, h1], [w1, h1], [w1, 0]]).reshape(-1, 1, 2)
    corners2 = np.float32([[0, 0], [0, h2], [w2, h2], [w2, 0]]).reshape(-1, 1, 2)
    warped_corners2 = cv2.perspectiveTransform(corners2, H)

    corners = np.concatenate((corners1, warped_corners2), axis=0)
    [xmin, ymin] = np.int32(corners.min(axis=0).ravel() - 0.5)
    [xmax, ymax] = np.int32(corners.max(axis=0).ravel() + 0.5)

    xRange = [xmin,xmax]
    yRange = [ymin,ymax]
    return xRange,yRange


def warpImage(im, H, xRange, yRange):
    
    T = np.eye(3)
    T[:2, 2] = [-xRange[0], -yRange[0]]
    H = T@H
    outSize = (xRange[1]-xRange[0], yRange[1]-yRange[0])
    mask = np.ones(im.shape[:2], dtype=np.uint8)*255
    imWarp = cv2.warpPerspective(im, H, outSize)
    maskWarp = cv2.warpPerspective(mask, H, outSize)
    return imWarp, maskWarp


def stitchImages(im1, im2, mask1, mask2):
    mask = mask1^mask2
    im2 = cv2.bitwise_and(im2,im2,mask = mask)
    panorama = (im1 + im2)
    return panorama



def get_kp(img,sift):
    """ 
    get sift keypoints and compute their descriptors, make sure to use cross checking.
    """
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    kp1, des1 = sift.detectAndCompute(img, None)
    return kp1, des1



def stitch_image(im1,im2):
    sift = cv2.SIFT_create()
    kp1, des1 = get_kp(im1,sift)
    kp2, des2 = get_kp(im2,sift) 
    H = estHomographyRANS(des1,des2,kp1,kp2)
    xRange,yRange = getRange(im1,im2,H)

    imWarp1, maskWarp1 = warpImage(im2, H, xRange, yRange)
    imwarp2,maskWarp2 = warpImage(im1, np.eye(3), xRange, yRange)
    panorama = stitchImages(imWarp1, imwarp2, maskWarp1, maskWarp2)
    return panorama


"""Expand your algorithm so it’s able to handle the situation where three or more images are taken
in a line."""

def stitch_images(images):
    panorama = stitch_image(images[0],images[1])
    for i in range(2,len(images)):
        panorama = stitch_image(panorama,images[i])
    return panorama


def Pi(coordinates):
    inhom = coordinates[:-1]/coordinates[-1]
    return inhom 

def PiInv(coordinate):
    """input: inhom coordinates
    returns: homogenous coordinates"""  
    if len(coordinate.shape)> 1 :
        w = np.ones((1,coordinate.shape[1]))
        hom = np.vstack((coordinate,w))

    else:

        w = 1
        hom = np.append(coordinate,w)

    return hom 


def project_points(K,R,t,Q):
    
    if len(Q.shape)>1:
        t = t.reshape(-1,1)
    
    p = K @ np.column_stack((R,t))
    p_h = p @ PiInv(Q)
    p_inh = Pi(p_h)
    return p_inh


def triangulate(qn,pn):
    """ 
    input:list of points qn and projection matrices Pn"""
    if len(qn) != len(pn):
        raise ValueError("Expected lists of equal length, len(Q)!=len(pn)")


    
    
    B_i = lambda P,q: np.array([  [P[2]*q[0] - P[0]],[P[2]*q[1] - P[1]]  ])

    B = np.hstack(([B_i(P_i,Pi(q_i)) for P_i, q_i in zip(pn,qn)]))

    B = B.reshape(len(pn)*2,4)
    U,S,VT = np.linalg.svd(B)

    Q = VT[-1,:]
    return Q


