import cv2 
import numpy as np 
import os 
import re 
import matplotlib.pyplot as plt 


def load_and_sort(path):

    imgs0 = []
    imgs1 = []
    files = os.listdir(path)
    files_0 = [file for file in files if 'frames0' in file]
    files_1 = [file for file in files if 'frames1' in file]
    files_0.sort(key=lambda x: int(x.split(".png")[0][8:]))
    files_1.sort(key=lambda x: int(x.split(".png")[0][8:]))
    for file0,file1 in zip(files_0,files_1):
        img0 = cv2.imread(path + file0)[:,:,::-1]
        img1 = cv2.imread(path + file1)[:,:,::-1]      
        imgs0.append(cv2.cvtColor(img0, cv2.COLOR_RGB2GRAY).astype(float)/255 )
        imgs1.append(cv2.cvtColor(img1,cv2.COLOR_RGB2GRAY).astype(float)/255 )
   
    return imgs0,imgs1 


def rectify_all_images(maps0,maps1,imgs0,imgs1):
    
    rectified_imgs0 = []
    rectified_imgs1 = []
    for im0,im1 in zip(imgs0,imgs1):
         rectified_imgs0.append(cv2.remap(im0, *maps0, cv2.INTER_LINEAR))
         rectified_imgs1.append(cv2.remap(im1, *maps1, cv2.INTER_LINEAR))
         
    return rectified_imgs0,rectified_imgs1
    
    

def plot_two_images(im0,im1,gray):
    if gray:
        cmap="gray" 
    else:
        cmap = None
    fig,ax = plt.subplots(1,2,figsize = (25,25)) 
    ax[0].imshow(im0,cmap=cmap) 
    ax[0].set_title("Image 1")
    
    ax[1].imshow(im1,cmap=cmap) 
    ax[1].set_title("Image 2")    
    plt.show()


def color_combine(im0, im1):
    # Create a color image where one image is red and the other is green
    red = np.zeros_like(im0)
    green = np.zeros_like(im1)
    blue = np.zeros_like(im1)
    color_image = np.stack([im0, green, blue], axis=-1)
    color_image1 = np.stack([red, im1, blue], axis=-1)
    combined = cv2.addWeighted(color_image, 0.5, color_image1, 0.5, 0)
    plt.imshow(combined)
    plt.title("Red-Green Color Combine")
    plt.show()


 
def draw_epipolar_lines(im0, im1, points):
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    ax[0].imshow(im0, cmap="gray")
    ax[0].set_title("Left Image")
    ax[1].imshow(im1, cmap="gray")
    ax[1].set_title("Right Image")

    # Draw a horizontal line across both images for each point
    for point in points:
        y = point[1]  # Assuming point is (x, y)
        ax[0].scatter(*point, color='red')  # Mark the point on the left image
        ax[0].axhline(y=y, color='red', linestyle='--')
        ax[1].axhline(y=y, color='red', linestyle='--')  # Same line on the right image

    plt.show()


def unwrap(imgs,n1):
    primary_images = imgs[2:17]
    fft_primary = np.fft.rfft(primary_images,axis=0)
    theta_primary = np.angle(fft_primary[1])
    secondary_images = imgs[17:26]
    fft_secondary = np.fft.rfft(secondary_images,axis=0)
    theta_secondary = np.angle(fft_secondary[1])
    #heterodyne principle 
    theta_cue = np.mod(theta_secondary - theta_primary,2*np.pi)
    o_primary = np.round(((n1 * theta_cue) - theta_primary) / (2 * np.pi))
    theta_est = np.mod((2*np.pi*o_primary+theta_primary)/(n1),2*np.pi)

    return theta_est 


def find_stereo_matches(theta0,theta1,mask0,mask1):
    rows,cols = theta0.shape 
    qs0 = []
    qs1 = [] 
    disparity = np.zeros((rows,cols),dtype=int)
    for i0 in range(rows):

        for j0 in range(cols): 
            best_match_j1 = None 
            if mask0[i0,j0]>0:
                
                min_phase_shift = float('inf')
                for j1 in range(cols):
                    phase_shift = abs(theta0[i0,j0] - theta1[i0,j1])
                    if min_phase_shift>phase_shift and mask1[i0,j1]>0:
                        min_phase_shift = phase_shift 
                        best_match_j1 = j1 
                if best_match_j1 is not None:
                    qs0.append((j0,i0))
                    qs1.append((best_match_j1,i0)) 
                    disparity[i0,j0] =   j0 - best_match_j1
    return qs0,qs1,disparity

    


def find_stereo_matches_fast(theta0, theta1, mask0, mask1):
    """
    Finds stereo matches between two rectified images based on phase matching.

    Parameters:
    - theta0 : ndarray, Phase image from camera 0.
    - theta1 : ndarray, Phase image from camera 1.
    - mask0  : ndarray, Mask indicating valid points in camera 0.
    - mask1  : ndarray, Mask indicating valid points in camera 1.

    Returns:
    - qs0        : list, Coordinates of matching points in camera 0.
    - qs1        : list, Coordinates of matching points in camera 1.
    - disparity  : ndarray, Disparity map where disparity[i0, j0] = j0 - j1 for matches.
    """
    rows, cols = theta0.shape
    qs0 = []
    qs1 = []
    disparity = np.zeros((rows, cols), dtype=int)  # Initialize disparity map

    for i0 in range(rows):
        # Find indices in current row i0 where both masks are True
        valid0_indices = np.where(mask0[i0])[0]
        valid1_indices = np.where(mask1[i0])[0]

        if valid0_indices.size > 0 and valid1_indices.size > 0:
            # Get the phase values for valid indices
            valid_phases0 = theta0[i0, valid0_indices]
            valid_phases1 = theta1[i0, valid1_indices]

            # Create a matrix of phase differences
            phase_diff_matrix = np.abs(valid_phases0[:, np.newaxis] - valid_phases1)

            # Find the index of the minimum phase difference for each valid0 index
            min_indices = np.argmin(phase_diff_matrix, axis=1)
            matched_j1s = valid1_indices[min_indices]

            # Store the coordinates of the matches
            qs0.extend([(j0, i0) for j0 in valid0_indices])
            qs1.extend([(j1, i0) for j1 in matched_j1s])

            # Calculate disparity
            for idx, j0 in enumerate(valid0_indices):
                disparity[i0, j0] = j0 - matched_j1s[idx]

    return qs0, qs1, disparity

# Example usage:
# theta0, theta1 are phase images from camera 0 and camera 1
# mask0, mask1 are binary masks indicating valid areas illuminated sufficiently
# Example data must be provided for theta0, theta1, mask0, mask1
# qs0, qs1, disparity = find_stereo_matches(theta0, theta1, mask0, mask1)
