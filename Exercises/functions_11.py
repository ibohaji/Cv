import numpy as np 
import cv2 
import matplotlib.pyplot as plt

def get_features(imgs):

    sift = cv2.SIFT_create(nfeatures = 2000)
    features = []

    for i in range(len(imgs)):
        
        im = imgs[i]

        kp, des = sift.detectAndCompute(im,None)

        kp0 = np.array([k.pt for k in kp])
        kp0 = kp0[:2000]
        des0 = des0[:2000]

        features.append((kp0, des0))

    return features




       

def get_matches():
    pass