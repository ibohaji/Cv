#!/usr/bin/env python
# coding: utf-8

# In[4]:


import itertools as it
import numpy as np 
import matplotlib.pyplot as plt 


def box3d(n=16):
    points = []
    N = tuple(np.linspace(-1, 1, n))
    for i, j in [(-1, -1), (-1, 1), (1, 1), (0, 0)]:
        points.extend(set(it.permutations([(i, )*n, (j, )*n, N])))
    return np.hstack(points)/2


# In[150]:


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


# In[221]:


def project_points(K,R,t,Q):
    
    if len(Q.shape)>1:
        t = t.reshape(-1,1)
    
    p = K @ np.column_stack((R,t))
    p_h = p @ PiInv(Q)
    p_inh = Pi(p_h)
    return p_inh



# In[228]:


Q = box3d() 
K = np.eye(3)
R = np.eye(3)
t = np.array([0,0,4])

p_h = project_points(K,R,t,Q)

x,y = p_h


# In[223]:



# Enable pretty-printing in the output




# In[224]:


x,y,z = box3d()
def plot3d(x,y,z):
    fig = plt.figure(figsize=(5, 5))
    ax = fig.add_subplot(111, projection='3d') # This creates a 3D plotting context
    ax.scatter(x, y, z) 

    # Setting labels for better understanding
    ax.set_xlabel('X Axis')
    ax.set_ylabel('Y Axis')
    ax.set_zlabel('Z Axis')
    plt.show()


plot3d(x,y,z)


# In[234]:


theta = np.radians(30)
R = np.array([[np.cos(theta),0,np.sin(theta)],[0,1,0],[-np.sin(theta),0,np.cos(theta)]])
p_h = project_points(K,R,t,Q)

x,y = p_h



