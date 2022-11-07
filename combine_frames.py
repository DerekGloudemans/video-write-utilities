import cv2
import numpy as np
import time

paths = [
        "/media/worklab/data_HDD/cv_data/video/data - test pole 6 cameras july 22/Jul_22_2019_12-05-07/Axis_Camera_10/cam_0_capture_002.avi",
        "/media/worklab/data_HDD/cv_data/video/data - test pole 6 cameras july 22/Jul_22_2019_12-05-07/Axis_Camera_12/cam_2_capture_002.avi",
        "/media/worklab/data_HDD/cv_data/video/data - test pole 6 cameras july 22/Jul_22_2019_12-05-07/Axis_Camera_13/cam_3_capture_002.avi",
        "/media/worklab/data_HDD/cv_data/video/data - test pole 6 cameras july 22/Jul_22_2019_12-05-07/Axis_Camera_14/cam_4_capture_002.avi",
        "/media/worklab/data_HDD/cv_data/video/data - test pole 6 cameras july 22/Jul_22_2019_12-05-07/Axis_Camera_15/cam_5_capture_002.avi",
        "/media/worklab/data_HDD/cv_data/video/data - test pole 6 cameras july 22/Jul_22_2019_12-05-07/Axis_Camera_16/cam_1_capture_002.avi"
        ]
paths = [
        
"/home/worklab/Documents/CV-detection/example_videos/110_foot_example_tracks/detect0.avi",
"/home/worklab/Documents/CV-detection/example_videos/110_foot_example_tracks/detect3.avi",
"/home/worklab/Documents/CV-detection/example_videos/110_foot_example_tracks/detect1.avi",
"/home/worklab/Documents/CV-detection/example_videos/110_foot_example_tracks/track0.avi",
"/home/worklab/Documents/CV-detection/example_videos/110_foot_example_tracks/track3.avi",
"/home/worklab/Documents/CV-detection/example_videos/110_foot_example_tracks/track1.avi"
]

paths = [
        
"/media/worklab/data_HDD/cv_data/video/110_foot_pole_test/Axis_Camera_10/cam_0_capture_000.avi",
"/media/worklab/data_HDD/cv_data/video/110_foot_pole_test/Axis_Camera_12/cam_2_capture_000.avi",
"/media/worklab/data_HDD/cv_data/video/110_foot_pole_test/Axis_Camera_13/cam_3_capture_000.avi",
"/media/worklab/data_HDD/cv_data/video/110_foot_pole_test/Axis_Camera_14/cam_4_capture_000.avi",
"/media/worklab/data_HDD/cv_data/video/110_foot_pole_test/Axis_Camera_15/cam_5_capture_000.avi",
"/media/worklab/data_HDD/cv_data/video/110_foot_pole_test/Axis_Camera_16/cam_1_capture_000.avi"
]

paths = [
    "/home/worklab/Documents/derek/i24-dataset-gen/camera_p1c1_track_outputs_3D.mp4",
    "/home/worklab/Documents/derek/i24-dataset-gen/camera_p1c2_track_outputs_3D.mp4",
    "/home/worklab/Documents/derek/i24-dataset-gen/camera_p1c3_track_outputs_3D.mp4",
    "/home/worklab/Documents/derek/i24-dataset-gen/camera_p1c4_track_outputs_3D.mp4",
    "/home/worklab/Documents/derek/i24-dataset-gen/camera_p1c5_track_outputs_3D.mp4",
    "/home/worklab/Documents/derek/i24-dataset-gen/camera_p1c6_track_outputs_3D.mp4"
    ]


file_out = "CIRCLES_6_camera_half_resolution.avi"
title_frame = None#  "/home/worklab/Desktop/110 Traffic Camera Pole Test.jpg"
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
frame_width = int(cap_list[0].get(3)*n_wide)
frame_height = int(cap_list[0].get(4)*n_high)

    
# opens VideoWriter object for saving video file if necessary
if file_out != None:
    out = cv2.VideoWriter(file_out,cv2.CAP_FFMPEG,cv2.VideoWriter_fourcc(*"MPEG"), 30, (2880,1080))

# add title frame for 2.5 seconds
if title_frame != None:
    title = cv2.imread(title_frame,1)
    for i in range(0,75):
        out.write(cv2.resize(title,(1920*3,1080*2)))
    
# read first frame from all captures
frames = []
#i = 0
for cap in cap_list:
    ret,frame = cap.read()
    #cv2.imwrite("align{}.jpg".format(i),frame)
    frame = cv2.resize(frame,(1920,1080))
    frames.append(frame)
    #i = i + 1
    

start = time.time()
frame_num = 0
while cap.isOpened():
    
    if ret:
        
        top_row = np.concatenate((frames[0],frames[1],frames[2]),axis = 1)
        bottom_row = np.concatenate((frames[3],frames[4],frames[5]),axis = 1)
        combined = np.concatenate((top_row,bottom_row),axis = 0)
        
        # save frame to file if necessary
        if file_out != None:
            combined = cv2.resize(combined, (2880, 1080))    
            out.write(combined)
        
        #summary statistics
        frame_num += 1
        print("\rFrame {}, {} fps".format(frame_num, frame_num / (time.time() - start)),end = "\r",flush = True)
                
        # get next frames
        frames = []
        for cap in cap_list:
            ret,frame = cap.read()
            if ret == False:
                break
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