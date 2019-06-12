import cv2
import matplotlib.pyplot as plt
import numpy as np
import time
import os
import signal
import subprocess
import multiprocessing as mp
from datetime import datetime

def record_stream(camera,directory,q):
    subdirectory = directory + '/' + camera['name']
    try:  
        os.mkdir(subdirectory)
        q.put (["Successfully created the directory %s " % subdirectory])
    except OSError:  
        q.put (["Creation of the directory %s failed" % subdirectory])

    base_name = subdirectory + '/' + 'capture_%03d.avi'
    command = ['ffmpeg', '-i', camera['address'], '-acodec', 'copy', '-vcodec', 'copy', '-r', '30.0','-map', '0', '-segment_time', '00:01:00', '-f', 'segment','-reset_timestamps', '1', base_name]
    p = subprocess.Popen(command,stdin=subprocess.PIPE)
    camera['pid'] = p.pid
    q.put(camera['pid'])
    
def test_fn(camera,directory,q):
    q.put("Done")
 
if __name__ == "__main__":
    
    # define cameras
    if True:
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
        camera_list.append(camera2)    
        
    # create overall directory for recordings
    directory = 'Recordings/'+ datetime.now().strftime('%m-%d-%Y_%H-%M-%S')
    try:  
        os.mkdir(directory)
    except OSError:  
        print ("Creation of the directory %s failed" % directory)
    else:  
        print ("Successfully created the directory %s " % directory)
    
    process_list = []
    subprocess_list = []
    manager = mp.Manager()
    q = manager.Queue()
    for camera in camera_list:
        # create separate process and call record_stream within process as subprocess
        args = {
                'camera': camera,
                'directory':directory,
                'q': q
        }
        p = mp.Process(target=test_fn, args=(*args,))
        p.start()
        process_list.append(p)
        if not q.empty():
            subprocess_list.append(q.get())
    input("Press Enter to stop capture...")
    for process in process_list:
        process.terminate()
        process.join()
        
    for pid in subprocess_list:
        os.kill(pid,15)
        
    print("All captures terminated.")