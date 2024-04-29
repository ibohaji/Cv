from re import M
import numpy as np 
import cv2 
import matplotlib.pyplot as plt
import os 


class visualOd:
    def __init__(self,folder,K):
    
        self.imgs = []
        self.keypoints = []
        self.descriptors = []
        self.matches = []
        self.K = K 
        self.pose  = []

        #Setting the position of the referencer frame.
        cam0_position = np.dot(np.eye(3).T, - np.zeros((3,1))) 
        self.position = [cam0_position]


        # Setting the initial pose 
        cam0_pose = self.get_transform(np.eye(3),np.zeros((1,3))) # Creating a transformation matrix 
        self.pose.append(cam0_pose)


        for filename in sorted(os.listdir(folder)):
            img = cv2.imread(os.path.join(folder,filename))[:,:,::-1]
            self.imgs.append(img)

    def compute_matches(self):
        """ 
        Gets the matches between the i'th and i+1 frame
        inputs: 
            - i: The i'th image/frame 
        output: matchesi_i+1
        """
        bf = cv2.BFMatcher(crossCheck = True) 

        for i in range(0,len(self.imgs)-1):
            match = bf.match(self.descriptors[i],self.descriptors[i+1])
            match = np.array([(m.queryIdx,m.trainIdx) for m in match])
            self.matches.append(match) 


    def get_transform(self,R,t):
        """ 
        Computes the transformation matrix
        ----------------------
        parameters: 
        -----------------------
        output: 
        """
        T = np.eye(4)
        
        # Set the rotation part
        T[0:3, 0:3] = R
        
        # Set the translation part
        T[0:3, 3] = t.reshape(1,-1)
        
        return T
            
    

    def compute_keypoints(self): 
        """ 
        input: the i'th image/frame
        output: all initial keypoints 
        """

        for i in range(len(self.imgs)):
                
            img = self.imgs[i]
            sift = cv2.SIFT_create(nfeatures = 2000 )
            kp,des = sift.detectAndCompute(img,None) 

            kp = np.array([k.pt for k in kp])

            kp = kp[:2000]
            des = des[:2000]

            self.keypoints.append(kp)
            self.descriptors.append(des) 

    def get_position(self,i): 
        """ 
        Computes the position of camera_i 

        """
        T = self.pose[i]
        R,t = self.decompose_transformation_matrix(T)
        position = R.T @ (-t) 

        return position 


    def recover_pose(self,i):
        """
        computes the essential matrix between the i'th and i+1 frame 
        input: the i'th frame/image
        output:The essential matrix E, the mask
        """

        kp1 = self.keypoints[i]
        kp2 = self.keypoints[i+1]

        matches01 = self.matches[i]
        matches12 = self.matches[i+1]

        E,mask = cv2.findEssentialMat(kp1[matches01[:,0]],kp2[matches01[:,1]],self.K,method=cv2.RANSAC)
        _,R,t,mask_pose = cv2.recoverPose(E,kp1[matches01[:,0]],kp2[matches01[:,1]],self.K)  # Recover the pose of the second camera 
        t = t
        combined_mask = (mask*mask_pose).astype(bool).flatten() 

        self.matches[i] = matches01[combined_mask]

        pose = self.get_transform(R,t) 
        self.pose.append(pose)
                

    def chain_feature_matches(self,i): 

        """
        Chains matches from image i to image i-2 through image i-1.
        
        Parameters:
        the i'th frame/image to chain 

        Returns:
        tuple: Tuple of three lists (points0, points1, points2) where each list contains the indices of matched points in the respective images.
        """

        matches01 = self.matches[i - 2]
        matches12 = self.matches[i - 1]  

        _, idx01, idx12 = np.intersect1d(matches01[:,1], matches12[:,0], return_indices=True)

        kp0 = self.keypoints[i-2]
        kp1 = self.keypoints[i-1]
        kp2 = self.keypoints[i]

        points0 = kp0[matches01[idx01,0]]
        points1 = kp1[matches01[idx01,1]]
        points2 = kp2[matches12[idx12,1]]

        return points0,points1,points2 

       
    def decompose_transformation_matrix(self,T):
        """
        Decompose a 4x4 transformation matrix to extract the rotation matrix R and translation vector t.

        Args:
        T (np.array): A 4x4 transformation matrix.

        Returns:
        tuple: A tuple containing the rotation matrix R and the translation vector t.
        """

        # Extract the rotation matrix R (top-left 3x3 submatrix of T)
        R = T[:3, :3]
        
        # Extract the translation vector t (first three elements of the fourth column of T)
        t = T[:3, 3]
        t = t.reshape(-1,1)
        
        return R, t

    def get_3D_objects(self,i,points0,points1): 
        """ 
        Obtains the 3D/world coordinates of the points between image i and i+1 
        -----------
        params:
            - i the i'th image/frame.
        outputs:
            - Q(ndarray) coordinates in homogenous coordinates
        """

        T0 = self.pose[i-2]
        T1 = self.pose[i-1]

        R0,t0 = self.decompose_transformation_matrix(T0)
        R1,t1 = self.decompose_transformation_matrix(T1)
    

        P0 = self.K @ np.hstack((R0,t0))
        P1 = self.K @ np.hstack((R1,t1)) 

        Q = cv2.triangulatePoints(P0,P1,points0.T,points1.T) #

        Q/= Q[3]  # Normalizing/dividing by w 
        Q = Q[:3].T.reshape(-1,1,3)

        return Q


    def estimatePose_pnp(self,imagePoints,Q):
        """ 
        Estimates the pose from 3D-2D point correspondance 
        Inputs: 
        - frame of the image to be estimated 
        - the 3D coordinate from the previous two frames?
        """
        _,rvec,tvec,inliers = cv2.solvePnPRansac(Q,imagePoints,self.K,distCoeffs = np.zeros(5)) 
        R = cv2.Rodrigues(rvec)[0]

        T = self.get_transform(R,tvec) # Storing the pose as a transformation matrix
        self.pose.append(T) # appending to the list of transformations
        return R,tvec,inliers
        
     
        






    


