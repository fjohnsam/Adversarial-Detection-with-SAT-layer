'''
Description: backpropagation operations for a convolutional neural network

Author: Alejandro Escontrela
Version: 1.0
Date: June 12th, 2018
'''

import numpy as np

from CNN.utils import *

#####################################################
############### Backward Operations #################
#####################################################

def satlayerBackward(image,dz_o,dz_istar,S,k,n,m,g_o,V,v_t,z_o,maxiter=100):
    #Computing dv_o #
    dv_o=dz_o.dot(v_t)/(3.14*np.sin(3.14*z_o))
    # Computing U #
    s_o=S[:,-1]
    u_o=np.zeroes(k,1)
    chi=np.zeroes(k,m)
    I_k=np.identity(k)
    v_o=V[:,-1]
    P_o=I_k-v_o.dot(np.transpose(v_o))

    i=0
    while i<maxiter:
        dg_o=chi.dot(s_o)-np.linalg.norm(s_o,ord=2).dot(u_o)-dv_o
        u_oprev=u_o
        u_o=-P_o.dot(dg_o)/np.linalg.norm(g_o,ord=1)
        chi+=(u_o - u_oprev).dot(np.transpose(s_o))
    # Computing dz_i and ds #
    v_lrand=np.random.rand(k,1)
    for i in range(n):
        s_i=S[:,i]
        z_i=[:,i][0]
        dvi_dzi=3.14*np.sin(3.14*z_i)*v_t+np.cos(3.14*z_i)*(I_k-v_t.dot(np.transpose(v_t))).dot(v_lrand)
        dz_i[:,i]=dz_istar[:,i]-np.transpose(dvi_dzi).dot(u_o.dot(np.transpose(s_o))).dot(s_i)

    U=np.zeroes(k,n)
    U=np.append(U,u_o,axis=1)
    ds=-np.transpose(u_o.dot(np.transpose(s_o))).dot(V)-(S.dot(np.transpose(V))).dot(U)

    return[ds,dz_i]

def convolutionBackward(dconv_prev, conv_in, filt, s):
    '''
    Backpropagation through a convolutional layer. 
    '''
    (n_f, n_c, f, _) = filt.shape
    (_, orig_dim, _) = conv_in.shape
    ## initialize derivatives
    dout = np.zeros(conv_in.shape) 
    dfilt = np.zeros(filt.shape)
    dbias = np.zeros((n_f,1))
    for curr_f in range(n_f):
        # loop through all filters
        curr_y = out_y = 0
        while curr_y + f <= orig_dim:
            curr_x = out_x = 0
            while curr_x + f <= orig_dim:
                # loss gradient of filter (used to update the filter)
                dfilt[curr_f] += dconv_prev[curr_f, out_y, out_x] * conv_in[:, curr_y:curr_y+f, curr_x:curr_x+f]
                # loss gradient of the input to the convolution operation (conv1 in the case of this network)
                dout[:, curr_y:curr_y+f, curr_x:curr_x+f] += dconv_prev[curr_f, out_y, out_x] * filt[curr_f] 
                curr_x += s
                out_x += 1
            curr_y += s
            out_y += 1
        # loss gradient of the bias
        dbias[curr_f] = np.sum(dconv_prev[curr_f])
    
    return dout, dfilt, dbias



def maxpoolBackward(dpool, orig, f, s):
    '''
    Backpropagation through a maxpooling layer. The gradients are passed through the indices of greatest value in the original maxpooling during the forward step.
    '''
    (n_c, orig_dim, _) = orig.shape
    
    dout = np.zeros(orig.shape)
    
    for curr_c in range(n_c):
        curr_y = out_y = 0
        while curr_y + f <= orig_dim:
            curr_x = out_x = 0
            while curr_x + f <= orig_dim:
                # obtain index of largest value in input for current window
                (a, b) = nanargmax(orig[curr_c, curr_y:curr_y+f, curr_x:curr_x+f])
                dout[curr_c, curr_y+a, curr_x+b] = dpool[curr_c, out_y, out_x]
                
                curr_x += s
                out_x += 1
            curr_y += s
            out_y += 1
        
    return dout
