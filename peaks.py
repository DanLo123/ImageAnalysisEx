# -*- coding: utf-8 -*-
"""
Created on Sun Jun 25 14:54:24 2023

@author: louis
"""
import tifffile as tf
import numpy as np
from PIL import Image
import cv2
from findpeaks import findpeaks


def localNorm(roi,window):
    roi[roi==0]=np.nan
    #iterate through each pixel of the image 
    norm_roi = np.zeros((roi.shape[0],roi.shape[1]))
    for i in range(roi.shape[0]): 
        if i>window and i<roi.shape[0]-(window+1): 
            for j in range(roi.shape[1]): 
                if j>window and j<roi.shape[1]-(window+1): 
                    if np.isfinite(roi[i,j]): 
                        #local window around pixel
                        local = roi[i-window:i+window+1,j-window:j+window+1]
                        #get the largest difference in the window
                        l_min = np.nanmin(local)
                        l_max = np.nanmax(local)
                        norm_roi[i,j] = (roi[i,j]-l_min)/(l_max-l_min)
                     
                    
                    else:
                        norm_roi[i,j] = 0
    return norm_roi

def enhance(roi): 
    #local intensity normalization for each pixel 
    window = 20
    #gaussian filter for noise
    roi = cv2.GaussianBlur(roi, (3, 3),0)
    norm_roi = localNorm(roi, window)
    
    return norm_roi
             
#iterate through ROI     
name = 'roi_'
num_cells=[]
for a in range(10):
    filename = name+str(a+1)+'.tif'

    #load the current roi
    roi = tf.imread(filename).astype(float)
       
    #local intensity normalization
    n_roi = enhance(roi)
        
    #find peaks 
    fp = findpeaks(method='mask',limit=0.3,denoise=None) 
    peaks = fp.fit(n_roi)
    
    p = peaks['Xdetect']
    
    roi_peak = n_roi*p 

    #Threshold Peaks    
    roi_peak[roi_peak<0.8] = 0
        
    #calculate unique peaks
    roi_peak = roi_peak*255
    analysis = cv2.connectedComponents(roi_peak.astype(np.uint8),cv2.CV_32S)
    num_cells.append(analysis[0])
total_cells = sum(num_cells)
print('There are '+str(total_cells)+' blue cell nuclei in the white pulp of the spleen')
