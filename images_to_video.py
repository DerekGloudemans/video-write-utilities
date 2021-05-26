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
     
     
    out = cv2.VideoWriter(os.path.join("/home/worklab/Desktop",'video.avi'),cv2.VideoWriter_fourcc(*'MPEG'), 30, size)
     
    for i in range(len(img_array)):
        out.write(img_array[i])
    out.release()
    
file = "/home/worklab/Data/cv/KITTI/data_tracking_image_2/training/image_02/0000"
file = "/home/worklab/Documents/derek/track_i24/output/temp_frames"
file = "/home/worklab/Documents/derek/LBT-count/vid"
im_to_vid(file)