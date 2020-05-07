'''
Description: forward operations for a convolutional neural network

Author: Alejandro Escontrela
Version: 1.0
Date: June 12th, 2018
'''
import numpy as np


#####################################################
################ Forward Operations #################
#####################################################

def satlayer(image,k,m,n_aux,S,v_t,maxiter=100):
    
    i_r,i_c = image.shape
    # Compute V_i #
    I_k=np.identity(k)
    V_i=np.random.rand((i_r,i_c))

    for col in range(i_c):
        z_i=image[:,col][0]
        V_i[:,col]=np.dot(-np.cos(3.14*z_i),v_t) + np.sin(3.14*z_i).dot(I_k-v_t.dot(np.transpose(v_t))).dot(V_i[:,col])

    # Compute V_o #
    V=V_i
    v_o=np.random.rand((k,1))
    V=np.append(V,v_o,axis=1)
    Omega=V.dot(np.transpose(S))

    i=0
    s_o=S[:,i_c+1]
    while i<maxiter:
        g_o=omega.dot(s_o)-np.linalg.norm(s_o,ord=2).dot(v_o)
        v_oprev=v_o
        v_o=-g_o/np.linalg.norm(g_o,ord=1)
        omega+=(v_o-v_oprev).dot(np.transpose(s_o))

    # Computing z_o #
    z_o=np.arccos((-np.transpose(v_o).dot(v_t))/3.14)

    return z_o


def convolution(image, filt, bias, s=1):
    '''
    Confolves `filt` over `image` using stride `s`
    '''
    (n_f, n_c_f, f, _) = filt.shape # filter dimensions
    n_c, in_dim, _ = image.shape # image dimensions
    
    out_dim = int((in_dim - f)/s)+1 # calculate output dimensions
    
    assert n_c == n_c_f, "Dimensions of filter must match dimensions of input image"
    
    out = np.zeros((n_f,out_dim,out_dim))
    
    # convolve the filter over every part of the image, adding the bias at each step. 
    for curr_f in range(n_f):
        curr_y = out_y = 0
        while curr_y + f <= in_dim:
            curr_x = out_x = 0
            while curr_x + f <= in_dim:
                out[curr_f, out_y, out_x] = np.sum(filt[curr_f] * image[:,curr_y:curr_y+f, curr_x:curr_x+f]) + bias[curr_f]
                curr_x += s
                out_x += 1
            curr_y += s
            out_y += 1
        
    return out

def maxpool(image, f=2, s=2):
    '''
    Downsample `image` using kernel size `f` and stride `s`
    '''
    n_c, h_prev, w_prev = image.shape
    
    h = int((h_prev - f)/s)+1
    w = int((w_prev - f)/s)+1
    
    downsampled = np.zeros((n_c, h, w))
    for i in range(n_c):
        # slide maxpool window over each part of the image and assign the max value at each step to the output
        curr_y = out_y = 0
        while curr_y + f <= h_prev:
            curr_x = out_x = 0
            while curr_x + f <= w_prev:
                downsampled[i, out_y, out_x] = np.max(image[i, curr_y:curr_y+f, curr_x:curr_x+f])
                curr_x += s
                out_x += 1
            curr_y += s
            out_y += 1
    return downsampled

def softmax(X):
    out = np.exp(X)
    return out/np.sum(out)

def categoricalCrossEntropy(probs, label):
    return -np.sum(label * np.log(probs))

