import cv2
import numpy as np
import os

def im_to_vid(directory): 
    img_array = []
    all_files = os.listdir(directory)
    all_files.sort()
    for filename in all_files:
        filename = os.path.join(directory, filename)
        img = cv2.imread(filename)
        height, width, layers = img.shape
        size = (width,height)
        img_array.append(img)
     
     
    out = cv2.VideoWriter(os.path.join(directory,'video.avi'),cv2.VideoWriter_fourcc(*'DIVX'), 15, size)
     
    for i in range(len(img_array)):
        out.write(img_array[i])
    out.release()
    
file = "/home/worklab/Documents/FAST-track/temp_output_ims"

im_to_vid(file)