import cv2
import matplotlib.pyplot as plt
import numpy as np
import time

from datetime import datetime
#%matplotlib inline  

# speed test for reading video in
if False:
    start_time = time.time()
    cap = cv2.VideoCapture("h_out0.avi")
            
             
    # Check if camera opened successfully
    assert cap.isOpened(), "Unable to read camera feed"
     
    
    
     
    while(True):
      ret, frame = cap.read()
     
      if ret == True: 
        
          pass
     
      # Break the loop
      else:
        break 
     
    # When everything done, release the video capture and video write objects
    cap.release()
    end_time = time.time()
    print("Time elapsed: {}".format(end_time - start_time))


if True:

    file_list = []
    file = {}
    file['path'] = '/home/worklab/Documents/CV-detection/data/20190422_102232_6D24.mkv'
    file['start_time'] = 20
    file['end_time'] = 120
    file_list.append(file)
    
    file = {}
    file['path'] = '/home/worklab/Documents/CV-detection/data/20190510_090616_25CE.mkv'
    file['start_time'] = 152
    file['end_time'] = 300
    file_list.append(file)
    
    file = {}
    file['path'] = '/home/worklab/Documents/CV-detection/data/20190510_091622_1A8F.mkv'
    file['start_time'] = 2
    file['end_time'] = 51
    file_list.append(file)
    
    file = {}
    file['path'] = '/home/worklab/Documents/CV-detection/data/capture_006.avi'
    file['start_time'] = 2
    file['end_time'] = 180
    file_list.append(file)
    
    file = {}
    file['path'] = '/home/worklab/Documents/CV-detection/data/capture_008.avi'
    file['start_time'] =0
    file['end_time'] = 160
    file_list.append(file)
    
    file = {}
    file['path'] = '/home/worklab/Documents/CV-detection/data/capture_009.avi'
    file['start_time'] = 60
    file['end_time'] = 225
    file_list.append(file)
    
    file = {}
    file['path'] = '/home/worklab/Documents/CV-detection/data/DJI_0010.MP4'
    file['start_time'] = 0
    file['end_time'] = 45
    file_list.append(file)
    
    i = 0
    for file in file_list:
        cap = cv2.VideoCapture(file['path'])
        
         
        # Check if camera opened successfully
        assert cap.isOpened(), "Unable to read camera feed"
         
        # Default resolutions of the frame are obtained.The default resolutions are system dependent.
        # We convert the resolutions from float to integer.
        frame_width = int(cap.get(3))
        frame_height = int(cap.get(4))
         
        # Define the codec and create VideoWriter object.The output is stored in 'outpy.avi' file.
        out = cv2.VideoWriter('traffic_{}.avi'.format(i),cv2.CAP_FFMPEG,cv2.VideoWriter_fourcc('H','2','6','4'), 30, (frame_width,frame_height))
    
     
        while(True):
          ret, frame = cap.read()
         
          if ret == True: 
            
            t = cap.get(0)/ 1000.0
            print(t)
            if t > file['start_time'] and t < file['end_time']:
                # Write the frame into the file 'output.avi'
                out.write(frame)

            
            # Display the resulting frame    
            cv2.imshow('frame',frame)
         
            # Press Q on keyboard to stop recording
            if cv2.waitKey(1) & 0xFF == ord('q'):
              break
           
            # end of segment of interest
            if t > file['end_time']:
                break
            
          # end of video
          else:
            break 
         
        # When everything done, release the video capture and video write objects
        cap.release()
        out.release()

        # Closes all the frames
        cv2.destroyAllWindows()
        
        i += 1 
        
        break
    
        
    cv2.destroyAllWindows()