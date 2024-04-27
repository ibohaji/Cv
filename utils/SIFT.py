from skimage import filters
import cv2 
import matplotlib.pyplot as plt 
import numpy as np 
import scipy 


def gaussian1DKernel(sigma):
    radius = int(4 * sigma)
    x = np.arange(-radius,radius+1).reshape(1,-1)
    
    g = 1/(np.sqrt(np.pi * sigma**2)) *np.exp(-x**2/(2*sigma**2))

    g /= np.sum(g) 
    
    return g


def apply_filter(im,sigma):

    g = gaussian1DKernel(sigma) 
    # Conolve with a gaussian 
    x_direction = cv2.filter2D(im,-1,g)
    i = cv2.filter2D(x_direction,-1,g.T) 
    return i

    

def scaleSpaced(im,sigma,n):
    """ 
    Naive implementation of the space scale pyramid
    -----
    inputs: 
    im: gray scale image
    sigma: standard deviation of the gaussian kernel  
    n: number of samples
    -----
    output: 
    pyramid: list of n scale space images 

    """
    pyramid = []
    sigmas = []

    for i in range(0,n-1):

        s = sigma*(2**i)
        img = apply_filter(im,s)
        pyramid.append(img)
        sigmas.append(s)
    
    return pyramid, sigmas



def differenceOfGaussians(im,sigma,n):
    
    """
    Generate a list of Difference of Gaussians (DoG) images from an input image.
    
    Parameters:
    - im (numpy.ndarray): The original image.
    - sigma (float): The standard deviation for the Gaussian blur.
    - n (int): The number of DoG images to generate.
    
    Returns:
    - List[numpy.ndarray]: A list of DoG images.
    
    """

        
    DoGs = [] 
    pyramid,sigmas = scaleSpaced(im,sigma,n+1)
    previous = pyramid[0]

    for i in range(1,len(pyramid)):
        current = pyramid[i]
        DoG =  (pyramid[i] -pyramid[i-1]) 
        DoGs.append(DoG) 
        previous = current 
    
    return DoGs,sigmas

def max_supr(r, x, y):
    # Ensure we do not go out of bounds
    if x == 0 or y == 0 or x == r.shape[0] - 1 or y == r.shape[1] - 1:
        return False
    return (r[x, y] > r[x+1, y]) and (r[x, y] >= r[x-1, y]) and (r[x, y] > r[x, y+1]) and (r[x, y] >= r[x, y-1])




def non_maximum_suppression(img, tau):
    N, M = img.shape
    suppressed = np.zeros_like(img)
    for x in range(1, N-1):
        for y in range(1, M-1):
            local_max = True
            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    if dx == 0 and dy == 0:
                        continue
                    if img[x, y] <= img[x+dx, y+dy]:
                        local_max = False
                        break
                if not local_max:
                    break
            if local_max and img[x, y] > tau:
                suppressed[x, y] = 1
    return suppressed



def detectBlobs(im,sigma,n,tau): 

    """
    Detects blobs in an image using the Difference of Gaussians (DoG) method with non-maximum suppression.

    Parameters:
    - im (np.array): The input image in which blobs are to be detected.
    - sigma (float): The standard deviation for the Gaussian kernel.
    - n (int): The number of scales to generate in the DoG scale space.
    - tau (float): The threshold value for DoG magnitudes to consider a pixel as a blob candidate.

    Returns:
    - blobs (list of tuples): A list where each tuple represents the coordinates of a detected blob.
    """

    N,M = im.shape
    DoGs,widths = differenceOfGaussians(im,sigma,n)
    n = len(DoGs) 
    MaxDoG = [cv2.dilate(abs(DoGs[i]), np.ones((3, 3))) for i in range(n)]
    blobs = []

    for i in range(0,n-1): 
        above = MaxDoG[i-1]
        img = DoGs[i]
        below = MaxDoG[i+1]
        scale = widths[i]

        suppressed = non_maximum_suppression(img, tau)

        for x in range(1,N-1):
            for y in range(1,M-1): 
                if (suppressed[x, y] == 1):
                    if img[x, y] > above[x, y] and img[x, y] > below[x, y]:
                        blobs.append((x, y,scale)) 
                        


    return blobs 



def visualize_blobs(blobs, im):
    # Visualization part (draw a circle for each blob)
    for x, y, scale in blobs:

        radius = int(scale*3)
        cv2.circle(im, (int(x), int(y)), radius, (10, 255, 255), 1)  

    cv2.imshow("Blob Detection", im)
    cv2.waitKey(0) 
    cv2.destroyAllWindows()  
