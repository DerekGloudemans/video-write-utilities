import cv2
import matplotlib.pyplot as plt
import numpy as np
import time
import os
import signal
import subprocess
from datetime import datetime

if True: # this block works
    command = ['ffmpeg', '-i', 'rtsp://root:worklab@192.168.1.10/axis-media/media.amp?framerate=30.0?streamprofile=preview', '-acodec', 'copy', '-vcodec', 'copy', '-r', '30.0', 'test_ff_mpeg_4.avi']
    #command = ['ffmpeg', '-i', 'rtsp://root:worklab@192.168.1.10/axis-media/media.amp?framerate=30.0?streamprofile=vlc', '-acodec', 'copy', '-vcodec', 'copy', '-r', '30.0', '-map', '0', '-segment_time', '00:00:05', '-f', 'segment', 'test_ff_mpeg_%03d.avi']

    #p = subprocess.Popen(command)
    p = subprocess.Popen(command,stdin=subprocess.PIPE)
    input("Press Enter to stop capture...")
    os.kill(p.pid, 15)

if False:
    # Axis camera parameters
    camera1 = {
        'name': 'Axis_Camera_1',
        'address': 'rtsp://root:worklab@192.168.1.10/axis-media/media.amp?framerate=30.0?streamprofile=vlc',
        'pid': None    
            }
    
    camera2 = {
        'name': 'Pelco_Camera_1',
        'address': 'rtsp://root:worklab@192.168.1.11/stream1',
        'pid': None 
            }
    camera_list = []
    camera_list.append(camera1)
    #camera_list.append(camera2)

    
    # create overall directory for recordings
    directory = 'Recordings/'+ datetime.now().strftime('%m-%d-%Y_%H-%M-%S')
    try:  
        os.mkdir(directory)
    except OSError:  
        print ("Creation of the directory %s failed" % directory)
    else:  
        print ("Successfully created the directory %s " % directory)
    
    
    for camera in camera_list:
        subdirectory = directory + '/' + camera['name']
        try:  
            os.mkdir(subdirectory)
        except OSError:  
            print ("Creation of the directory %s failed" % subdirectory)
        else:  
            print ("Successfully created the directory %s " % subdirectory)
        
        base_name = subdirectory + '/' + 'capture_%03d.avi'
        command = ['ffmpeg', '-i', camera['address'], '-acodec', 'copy', '-vcodec', 'copy', '-r', '30.0','-map', '0', '-segment_time', '00:01:00', '-f', 'segment','-reset_timestamps', '1', base_name]
        p = subprocess.Popen(command,stdin=subprocess.PIPE)
        camera['pid'] = p.pid
        print("Started recording on {}.".format(camera['name']))


    input("Press Enter to stop capture...")
    for camera in camera_list:
        os.kill(camera['pid'],15)
   # print("All captures terminated.")
