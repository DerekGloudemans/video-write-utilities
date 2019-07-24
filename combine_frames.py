import cv2
import numpy as np
import time

paths = [
        "/media/worklab/data_HDD/cv_data/video/data - test pole 6 cameras july 22/Jul_22_2019_12-05-07/Axis_Camera_10/cam_0_capture_003.avi",
        "/media/worklab/data_HDD/cv_data/video/data - test pole 6 cameras july 22/Jul_22_2019_12-05-07/Axis_Camera_12/cam_2_capture_003.avi",
        "/media/worklab/data_HDD/cv_data/video/data - test pole 6 cameras july 22/Jul_22_2019_12-05-07/Axis_Camera_13/cam_3_capture_003.avi",
        "/media/worklab/data_HDD/cv_data/video/data - test pole 6 cameras july 22/Jul_22_2019_12-05-07/Axis_Camera_14/cam_4_capture_003.avi",
        "/media/worklab/data_HDD/cv_data/video/data - test pole 6 cameras july 22/Jul_22_2019_12-05-07/Axis_Camera_15/cam_5_capture_003.avi",
        "/media/worklab/data_HDD/cv_data/video/data - test pole 6 cameras july 22/Jul_22_2019_12-05-07/Axis_Camera_16/cam_1_capture_003.avi"
        ]
file_out = "combined_1.avi"
show = False

# open capture devices to read video files
cap_list = []
for file_in in paths:
    cap = cv2.VideoCapture(file_in)
    assert cap.isOpened(), "Cannot open file \"{}\"".format(file_in)
    
    cap_list.append(cap)
    
# determine size of combined frame (assumed that all frames are the same size)
cam_num = len(cap_list)
n_wide = 3
n_high = (cam_num-1) // 3 + 1
frame_width = int(cap_list[0].get(3)*n_wide /2.0)
frame_height = int(cap_list[0].get(4)*n_high /2.0)

    
# opens VideoWriter object for saving video file if necessary
if file_out != None:
    out = cv2.VideoWriter(file_out,cv2.CAP_FFMPEG,cv2.VideoWriter_fourcc('H','2','6','4'), 30, (frame_width,frame_height))


# read first frame from all captures
frames = []
for cap in cap_list:
    ret,frame = cap.read()
    frame = cv2.resize(frame,(1920,1080))
    frames.append(frame)

start = time.time()
frame_num = 0
while cap.isOpened():
    
    if ret:
        
        top_row = np.concatenate((frames[0],frames[1],frames[2]),axis = 1)
        bottom_row = np.concatenate((frames[3],frames[4],frames[5]),axis = 1)
        combined = np.concatenate((top_row,bottom_row),axis = 0)
        
        # save frame to file if necessary
        if file_out != None:
            out.write(combined)
        
        
        #summary statistics
        frame_num += 1
        print("FPS of the video is {:5.2f}".format( frame_num / (time.time() - start)))
                
        # get next frames
        frames = []
        for cap in cap_list:
            ret,frame = cap.read()
            frame = cv2.resize(frame,(1920,1080))
            frames.append(frame)
        
        # output frame
        if show:
            combined = cv2.resize(combined, (2880, 1080))               
            cv2.imshow("frame", combined)
            key = cv2.waitKey(1)
            if key & 0xFF == ord('q'):
                break
            continue
        
 
            
    else:
        break
    
# close all resources used
for cap in cap_list:
    cap.release()
cv2.destroyAllWindows()
try:
    out.release()
except:
    pass

print("Video combination finished.")