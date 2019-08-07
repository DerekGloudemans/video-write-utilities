import cv2
import time
from os import mkdir

"""
makes a directory with each frame of the input video file as a separate image. 
Useful to get rid of the overhead of video decoding and encoding during detection etc.
"""
def video_to_images(video_file,out_directory = "temp"):
    # open up a videocapture object
    cap = cv2.VideoCapture(video_file)
    # verify file is opened
    assert cap.isOpened(), "Cannot open file \"{}\"".format(video_file)
    
    #create directory for new images
    try:  
        mkdir(out_directory)
    except OSError:  
        print ("Creation of the directory %s failed" % out_directory)
    else:  
        print ("Successfully created the directory %s " % out_directory)
    
    start = time.time()
    frame_num = 0

    
    # get first frame
    ret, frame = cap.read()
    
    while ret:  

        cv2.imwrite(out_directory + "/{}.png".format(frame_num),frame)
        print("On frame: {}, FPS: {:5.2f}".format(frame_num, 1.0 / (time.time() - start)))
        frame_num += 1
        start = time.time()
        # get second frame
        ret, frame = cap.read()
        if frame_num > 30*120:
            cap.release()
            break
        

video_file = '/media/worklab/data_HDD/cv_data/video/traffic_assorted/traffic_0.avi'
out_directory = '/media/worklab/data_HDD/cv_data/video/traffic_assorted/traffic_0_frames'
video_to_images(video_file,out_directory)

class ImDirectoryReader():
    def __init__(self,path):
        self.path = path
        self.frame = 0
    
    def get(self):
        try:
            im = cv2.imread(self.path+ "/{}.png".format(self.frame))
            self.frame += 1
            return True, im
        except:
            return False, None
        
    def get_frame(self,frame_num):
        try:
            return True, cv2.imread(self.path+ "/{}.png".format(frame_num))
        except:
            return False, None
        