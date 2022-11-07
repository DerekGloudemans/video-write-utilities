import os
import subprocess
from datetime import datetime
import ctypes

# this line was successful
#ffmpeg -rtsp_transport tcp -use_wallclock_as_timestamps 1 -i rtsp://root:worklab@192.168.1.10/axis-media/media.amp?framerate=30.0?streamprofile=vlc -codec:v:0 copy -r 30.0 -segment_time 00:00:05 -f segment -reset_timestamps 1 test_ff_mpeg_%03d.avi


def test(): # this block works
    command = ['ffmpeg', '-i', 'rtsp://root:worklab@192.168.1.10/axis-media/media.amp?framerate=30.0?streamprofile=vlc', '-c:a', 'copy', '-c:v', 'copy', '-r', '30.0', 'test_ff_mpeg_1.avi']
    #command = ['ffmpeg', '-i', 'rtsp://root:worklab@192.168.1.10/axis-media/media.amp?framerate=30.0?streamprofile=vlc', '-acodec', 'copy', '-vcodec', 'copy', '-r', '30.0', '-map', '0', '-segment_time', '00:00:05', '-f', 'segment', 'test_ff_mpeg_%03d.avi']

    #p = subprocess.Popen(command)
    p = subprocess.Popen(command,stdin=subprocess.PIPE)
    print("PID: {}".format(int(p.pid)))
    input("Press Enter to stop capture...")
    #os.kill(p.pid, 15)
    ctypes.windll.kernel32.TerminateProcess(int(p._handle), -1)


def get_camera_list(mode = None):

    # Axis camera parameters
    camera1 = {
        'name': 'Axis_Camera_0',
        'address': 'rtsp://root:worklab@192.168.1.10/axis-media/media.amp?framerate=30.0?streamprofile=vlc',
        'pid': None,
        'num':0
            }
    
    # Pelco camera parameters
    camera2 = {
        'name': 'Pelco_Camera_1',
        'address': 'rtsp://root:worklab@192.168.1.11/stream1',
        'pid': None,
        'num': 1
            }
    
    camera_list = []
    camera_list.append(camera1)
    camera_list.append(camera2)
    
    if mode == "6-stream-test":
        # Axis camera parameters
        camera3 = {
            'name': 'Axis_Camera_2',
            'address': 'rtsp://root:worklab@192.168.1.10/axis-media/media.amp?framerate=30.0?streamprofile=vlc',
            'pid': None,
            'num':2
                }
        camera4 = {
            'name': 'Pelco_Camera_3',
            'address': 'rtsp://root:worklab@192.168.1.11/stream1',
            'pid': None,
            'num': 3
                }
        camera5 = {
            'name': 'Axis_Camera_4',
            'address': 'rtsp://root:worklab@192.168.1.10/axis-media/media.amp?framerate=30.0?streamprofile=vlc',
            'pid': None,
            'num':4
                }
        
        camera6 = {
            'name': 'Pelco_Camera_5',
            'address': 'rtsp://root:worklab@192.168.1.11/stream1',
            'pid': None,
            'num': 5
                }
        camera_list.append(camera3)
        camera_list.append(camera4)
        camera_list.append(camera5)
        camera_list.append(camera6)
    return camera_list


#############################3 Begin main code ################################
   
camera_list = get_camera_list()
segment_length = "00:05:00"

# create overall directory for recordings
directory = 'Recordings/'+ datetime.now().strftime('%b_%d_%Y_%H-%M-%S')
try:  
    os.mkdir(directory)
except OSError:  
    print ("Creation of the directory %s failed" % directory)
else:  
    print ("Successfully created the directory %s " % directory)


for camera in camera_list:
    # create subdirectory per camera
    subdirectory = directory + '/' + camera['name']
    try:  
        os.mkdir(subdirectory)
    except OSError:  
        print ("Creation of the directory %s failed" % subdirectory)
    else:  
        print ("Successfully created the directory %s " % subdirectory)
    
    # define file name format
    base_name = subdirectory + '/' + 'cam_{}_capture_%03d.avi'.format(camera['num'])
    
    command = 'ffmpeg -rtsp_transport tcp  -use_wallclock_as_timestamps 1 -i {} \
    -codec:v:0 copy -r 30.0 -segment_time {} -f segment \
    -reset_timestamps 1 {}'.format(camera['address'],segment_length,base_name)
    
    p = subprocess.Popen(command,stdin=subprocess.PIPE,shell=False)
    camera['pid'] = p.pid
    camera['p'] = p
    print("Started recording on {}.".format(camera['name']))


input("Press Enter to stop capture...")
for camera in camera_list:
    #os.kill(camera['pid'],15) # I think this works for linux
    ctypes.windll.kernel32.TerminateProcess(int(camera['p']._handle), -1) # for windows

print("All captures terminated.")