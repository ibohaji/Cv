from re import M
import numpy as np 
import cv2 
import matplotlib.pyplot as plt
import os 
import plotly.graph_objects as go

class VisualOdometry:
    def __init__(self,folder,K):
    
        self.imgs = [cv2.imread(os.path.join(folder, f))[:, :, ::-1] for f in sorted(os.listdir(folder))]
        self.keypoints = []
        self.descriptors = []
        self.matches = []
        self.K = K 
        self.feature_tracks = {}
        self.pose = [self.get_transform(np.eye(3), np.zeros(3))]
        #Setting the position of the referencer frame.
        cam0_position = np.dot(np.eye(3).T, - np.zeros((3,1))) 
        self.position = [cam0_position]


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

    @staticmethod
    def get_transform(R,t):
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
        T[0:3, 3] = t.flatten()
    
        return T
            
        


    def compute_keypoints(self): 
        """ 
        input: the i'th image/frame
        output: all initial keypoints 
        """

        sift = cv2.SIFT_create(nfeatures = 4000 )
        for img in self.imgs:
            
            kp, des = sift.detectAndCompute(img, None)
            self.keypoints.append(np.array([k.pt for k in kp])[:4000])
            self.descriptors.append(des[:4000]) 



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

        E,mask = cv2.findEssentialMat(kp1[matches01[:,0]],kp2[matches01[:,1]],self.K,method=cv2.RANSAC,prob=0.999) # Compute the essential matrix between the two frames
        _,R,t,mask_pose = cv2.recoverPose(E,kp1[matches01[:,0]],kp2[matches01[:,1]],self.K,mask=mask)  # Recover the pose of the second camera 
        combined_mask = (mask*mask_pose).astype(bool).flatten() 
        self.matches[i] = matches01[combined_mask]

        pose = self.get_transform(R,t) 
        self.pose.append(pose)

    """ Match Features Across Frames: Use feature matching algorithms, like FLANN or BFMatcher in OpenCV, to find correspondences of the same feature across multiple frames. You should already have a list of matches between consecutive frames (from exercises 11.1 to 11.5). Now you need to link these matches across the entire sequence of images.
"""

    def match_features_across_frames(self):
        """
        Matches features across all frames in the sequence.
        """
        bf = cv2.BFMatcher(crossCheck=True)
        for i in range(1, len(self.imgs)):
            matches = []
            for j in range(i):
                matches_ij = bf.match(self.descriptors[j], self.descriptors[i])
                matches_ij = np.array([(m.queryIdx, m.trainIdx) for m in matches_ij])
                matches.append(matches_ij)
            self.matches.append(matches)

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
        points1 = kp1[matches12[idx12,0]]
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


    def get_3D_points(self,i): 
        """ 
        Obtains the 3D points from the i'th frame, by triangulating the points from image i to all the images for which all features are detectable 
        """
        matches = self.matches[i]
        points = [] 
        P = [] 

        for i,match in enumerate(matches):
            if match is not None: 
                kp_i = self.keypoints[i]
                point = kp_i[match[0]]
                T = self.pose[i]
                R,t = self.decompose_transformation_matrix(T)

                P.append(self.K @ np.hstack((R,t)))
                points.append(point)
        Q = triangulate(P,points)

        return Q

# kp0[matches01[idx01,0]]

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

        Q = cv2.triangulatePoints(P0,P1,points0.T,points1.T) 
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
    
    
    @staticmethod 
    def triangulate(qn,pn):
        """ 
        input:
        - qn: list of points
        - pn: list of projection matrices
        ----------------------
        output:
        - Q: 3D points
        """
        if len(qn) != len(pn):
            raise ValueError("Expected lists of equal length, len(Q)!=len(pn)")

        B_i = lambda P,q: np.array([  [P[2]*q[0] - P[0]],[P[2]*q[1] - P[1]]  ])

        B = np.hstack(([B_i(P_i,Pi(q_i)) for P_i, q_i in zip(pn,qn)]))

        B = B.reshape(len(pn)*2,4)
        U,S,VT = np.linalg.svd(B)

        Q = VT[-1,:]
        return Q



    @staticmethod        
    def plot_3d_points_and_cameras(Q, camera_positions, inliers):
        
        """
        Plots 3D points and camera positions using Plotly.

        :param Q: Numpy array of 3D points, shape (3, N).
        :param camera_positions: List of numpy arrays, each with shape (3,) representing camera positions.
        :param inliers: Array or list of indices for the inlier points to be plotted.
        """
        inlier_Q = Q[inliers.flatten()]  # Select inliers and ensure the array is properly shaped
        inlier_Q = inlier_Q.reshape(-1, 3)

        fig = go.Figure()

        # Add the triangulated 3D inlier points to the plot
        fig.add_trace(go.Scatter3d(
            x=inlier_Q[:, 0],  # X coordinates
            y=inlier_Q[:, 1],  # Y coordinates
            z=inlier_Q[:, 2],  # Z coordinates
            mode='markers',
            marker=dict(
                size=2,
                color='blue',  # Points color
                opacity=0.8
            ),
            name='Triangulated Points'
        ))

        # Loop through the list of camera positions and add each to the plot
        for i, pos in enumerate(camera_positions):
            fig.add_trace(go.Scatter3d(
                x=pos[0],
                y=pos[1],
                z=pos[2],
                mode='markers',
                marker=dict(
                    size=5,
                    color='green' if i == 0 else 'red',  # Use different colors for clarity, customize as needed
                    opacity=0.8
                ),
                name=f'Camera {i} Position'
            ))

        # Configure the layout
        fig.update_layout(
            title='3D Point Cloud and Camera Positions',
            scene=dict(
                xaxis_title='X Axis',
                yaxis_title='Y Axis',
                zaxis_title='Z Axis',
                xaxis=dict(showgrid=True, zeroline=False),
                yaxis=dict(showgrid=True, zeroline=False),
                zaxis=dict(showgrid=True, zeroline=False),
                aspectmode='data'
            ),
            margin=dict(l=0, r=0, b=0, t=0)
        )

        # Show the figure
        fig.show()