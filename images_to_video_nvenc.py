# Copyright 2019 NVIDIA Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

# Starting from Python 3.8 DLL search policy has changed.
# We need to add path to CUDA DLLs explicitly.
import numpy as np
from enum import Enum
import PyNvCodec as nvc
import sys
import os
import logging
import PIL

logger = logging.getLogger(__file__)

if os.name == "nt":
    # Add CUDA_PATH env variable
    cuda_path = os.environ["CUDA_PATH"]
    if cuda_path:
        os.add_dll_directory(cuda_path)
    else:
        logger.error("CUDA_PATH environment variable is not set.")
        logger.error("Can't set CUDA DLLs search path.")

    # Add PATH as well for minor CUDA releases
    sys_path = os.environ["PATH"]
    if sys_path:
        paths = sys_path.split(";")
        for path in paths:
            if os.path.isdir(path):
                os.add_dll_directory(path)
    else:
        logger.error("PATH environment variable is not set.")




def encode(gpuID, decFilePath, encFilePath, width, height):
    encFile = open(encFilePath, "wb")
    res = str(width) + 'x' + str(height)

    nvEnc = nvc.PyNvEncoder({'preset': 'P5', 'tuning_info': 'high_quality', 'codec': 'h264',
                             'profile': 'high', 's': res, 'bitrate': '10M'}, gpuID)

    nv12FrameSize = int(nvEnc.Width() * nvEnc.Height() * 3 / 2)
    encFrame = np.ndarray(shape=(0), dtype=np.uint8)

    #Number of frames we've sent to encoder
    framesSent = 0
    #Number of frames we've received from encoder
    framesReceived = 0
    #Number of frames we've got from encoder during flush.
    #This number is included in number of received frames.
    #We use separate counter to check if encoder receives packets one by one
    #during flush.
    framesFlushed = 0
    
    all_files = os.listdir(decFilePath)
    all_files.sort()
    
    total_num_frames = 0
    for file in all_files:
        file = os.path.join(decFilePath,file)
        
        #open file
        rawFrame = np.asarray(PIL.Image.open(file)) 
        rawFrame = (rawFrame*255).astype(np.uint8)
        
        if not (rawFrame.size):
            print('No more input frames')
            break

        success = nvEnc.EncodeSingleFrame(rawFrame, encFrame, sync=False)
        framesSent += 1

        if(success):
            encByteArray = bytearray(encFrame)
            encFile.write(encByteArray)
            framesReceived += 1
            

    #Encoder is asynchronous, so we need to flush it
    while True:
        success = nvEnc.FlushSinglePacket(encFrame)
        if (success) and (framesReceived < framesSent):
            encByteArray = bytearray(encFrame)
            encFile.write(encByteArray)
            framesReceived += 1
            framesFlushed += 1
        else:
            break

    print(framesReceived, '/', total_num_frames,
          ' frames encoded and written to output file.')
    print(framesFlushed, ' frame(s) received during encoder flush.')


if __name__ == "__main__":

    gpuID = 0
    file = "/home/derek/Desktop/temp_frames"
    encFilePath = "/home/derek/Desktop/temp.mp4"
    width = 2160
    height = 3840
    encode(gpuID, file, encFilePath, width, height)

