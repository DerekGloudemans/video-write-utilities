"""
Usage - Tiles all videos in directory together, trimming to the length of the shortest video
args - directory_path - path to video directory

By assumption, files all contain an I24-MOTION camera identifier (e.g. PXXCXX)
and will be aranged in order of decreasing P, increasing C (e.g. into-town ordering)
"""

d = "/home/derek/Data/MOTION_homography_10_2022"
encFilePath = "/home/derek/Desktop/mega_output_test.mp4"
scale = 2
target_size = (1920*scale,1080*scale)

import sys,os
import numpy as np
import torch
import PyNvCodec as nvc
import PytorchNvCodec as pnvc
import cv2
import skvideo.io

class cconverter:
    """
    Colorspace conversion chain.
    """

    def __init__(self, width: int, height: int, gpu_id: int):
        self.gpu_id = gpu_id
        self.w = width
        self.h = height
        self.chain = []

    def add(self, src_fmt: nvc.PixelFormat, dst_fmt: nvc.PixelFormat) -> None:
        self.chain.append(nvc.PySurfaceConverter(
            self.w, self.h, src_fmt, dst_fmt, self.gpu_id))

    def run(self, src_surface: nvc.Surface) -> nvc.Surface:
        surf = src_surface
        cc = nvc.ColorspaceConversionContext(nvc.ColorSpace.BT_601,
                                             nvc.ColorRange.MPEG)

        for cvt in self.chain:
            surf = cvt.Execute(surf, cc)
            if surf.Empty():
                raise RuntimeError('Failed to perform color conversion')

        return surf.Clone(self.gpu_id)


def tensor_to_surface(img_tensor: torch.tensor, gpu_id: int) -> nvc.Surface:
    """
    Converts cuda float tensor to planar rgb surface.
    """
    if len(img_tensor.shape) != 3 and img_tensor.shape[0] != 3:
        raise RuntimeError('Shape of the tensor must be (3, height, width)')

    tensor_w, tensor_h = img_tensor.shape[2], img_tensor.shape[1]
    img = torch.clamp(img_tensor, 0.0, 1.0)
    img = torch.multiply(img, 255.0)
    img = img.type(dtype=torch.cuda.ByteTensor)

    surface = nvc.Surface.Make(
        nvc.PixelFormat.RGB_PLANAR, tensor_w, tensor_h, gpu_id)
    surf_plane = surface.PlanePtr()
    pnvc.TensorToDptr(img, surf_plane.GpuMem(),
                      surf_plane.Width(),
                      surf_plane.Height(),
                      surf_plane.Pitch(),
                      surf_plane.ElemSize())

    return surface 





directory = d#sys.argv[1]

# get list of video files
files = os.listdir(directory)

sorted_files = []
sorted_names = []
# sort in order of decreasing pole, increasing camera
for p in range(48,0,-1):
    if p == 47:
        continue
    for c in range(1,13,1):
        cname = "P{}C{}".format(str(p).zfill(2),str(c).zfill(2))
        for file in files:
            if cname in file:
                sorted_files.append(file)
                sorted_names.append(cname)
                break
    
    # if len(sorted_files) == 36:
    #     break
            
print(sorted_files)
n = len(sorted_files)


# get appropriate tile size
square = int(np.ceil(n**0.5))


all_loaders = []
resize = (int(target_size[1]//square),int(target_size[0]//square))

# intialize a decoder for each file
for fidx,file in enumerate(sorted_files):
    file = os.path.join(directory,file)
    
    gpuID = fidx % torch.cuda.device_count()
    
    print("Initializing loader for file {}".format(file))
    
    nvDec = nvc.PyNvDecoder(file, gpuID)


    
    # initialize extra goodies once
    target_h, target_w = nvDec.Height(), nvDec.Width()
        
    to_rgb = nvc.PySurfaceConverter(nvDec.Width(), nvDec.Height(), nvc.PixelFormat.NV12, nvc.PixelFormat.RGB, gpuID)
    to_planar = nvc.PySurfaceConverter(nvDec.Width(), nvDec.Height(), nvc.PixelFormat.RGB, nvc.PixelFormat.RGB_PLANAR, gpuID)
    
    cspace, crange = nvDec.ColorSpace(), nvDec.ColorRange()
    if nvc.ColorSpace.UNSPEC == cspace:
        cspace = nvc.ColorSpace.BT_601
    if nvc.ColorRange.UDEF == crange:
        crange = nvc.ColorRange.MPEG
    cc_ctx = nvc.ColorspaceConversionContext(cspace, crange)
    
    # if fidx > 20:
    #     break
    
    all_loaders.append([file,gpuID,nvDec,target_h,target_w,to_rgb,to_planar,cspace,crange,cc_ctx])
    
    
    
# INIT encoder
encFile = open(encFilePath, "wb")
res = str(target_size[0]) + 'x' + str(target_size[1])

nvEnc = nvc.PyNvEncoder({'preset': 'P5', 'tuning_info': 'high_quality', 'codec': 'h264',
                         'profile': 'high', 's': res, 'bitrate': '10M'}, 0)

nv12FrameSize = int(nvEnc.Width() * nvEnc.Height())
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
 
# to_rgb = cconverter(w, h, gpu_id)
# to_rgb.add(nvc.PixelFormat.NV12, nvc.PixelFormat.YUV420)
# to_rgb.add(nvc.PixelFormat.YUV420, nvc.PixelFormat.RGB)
# to_rgb.add(nvc.PixelFormat.RGB, nvc.PixelFormat.RGB_PLANAR)
    
to_nv12 = cconverter(target_size[0], target_size[1], 0)
to_nv12.add(nvc.PixelFormat.RGB_PLANAR, nvc.PixelFormat.RGB)
to_nv12.add(nvc.PixelFormat.RGB, nvc.PixelFormat.YUV420)
to_nv12.add(nvc.PixelFormat.YUV420, nvc.PixelFormat.NV12)
   
encDevice = torch.device("cuda:0")


font = cv2.FONT_HERSHEY_SIMPLEX
magic_scale = 10
fontscale = magic_scale * resize[0]/3840
name_tensors = []
shift = int(9*magic_scale * resize[0]/3840)
for name in sorted_names:
    t = np.zeros([resize[0],resize[1],3],dtype = float)
    cv2.putText(t,name,(shift,3*shift),font,fontscale,(255,255,255),4)
    cv2.putText(t,name,(shift,3*shift),font,fontscale,(-255,-255,-255),2)

    t=torch.from_numpy(t).permute(2,0,1)
    name_tensors.append(t)
    
# one loop =one frame from each camera
frame_counter = 0
while True:
    this_frame = []
    print("On frame {} ----> framesSent: {}   framesRecieved: {}   framesFlushed: {}".format(frame_counter,framesSent,framesReceived,framesFlushed))
   
    for lidx,loader in enumerate(all_loaders):
        # get first frame
        [file,gpuID,nvDec,target_h,target_w,to_rgb,to_planar,cspace,crange,cc_ctx] = loader
        
        pkt = nvc.PacketData()                    
        rawSurface = nvDec.DecodeSingleSurface(pkt)
        ts = pkt.pts /10e8
        
        # get frames from one file
        if rawSurface.Empty():
            break
       
        
        # Obtain NV12 decoded surface from decoder;
        #raw_surface = nvDec.DecodeSingleSurface(pkt)
        if rawSurface.Empty():
            break

        # Convert to RGB interleaved;
        rgb_byte = to_rgb.Execute(rawSurface, cc_ctx)
    
        # Convert to RGB planar because that's what to_tensor + normalize are doing;
        rgb_planar = to_planar.Execute(rgb_byte, cc_ctx)
    
        # likewise, end of video file
        if rgb_planar.Empty():
            break
        
        # Create torch tensor from it and reshape because
        # pnvc.makefromDevicePtrUint8 creates just a chunk of CUDA memory
        # and then copies data from plane pointer to allocated chunk;
        surfPlane = rgb_planar.PlanePtr()
        surface_tensor = pnvc.makefromDevicePtrUint8(surfPlane.GpuMem(), surfPlane.Width(), surfPlane.Height(), surfPlane.Pitch(), surfPlane.ElemSize())
        surface_tensor.resize_(3, target_h,target_w)
        
        
        try:
            surface_tensor = torch.nn.functional.interpolate(surface_tensor.unsqueeze(0),resize).squeeze(0)
        except:
            raise Exception("Surface tensor shape:{} --- resize shape: {}".format(surface_tensor.shape,resize))
    
        # add camera name overlay
        # TODO!!!
        surface_tensor = surface_tensor.type(dtype=torch.cuda.FloatTensor)
        surface_tensor = surface_tensor.cpu()
        surface_tensor += name_tensors[lidx]
        surface_tensor = torch.clamp(surface_tensor,min = 0,max = 255)
        surface_tensor = surface_tensor.type(dtype=torch.ByteTensor)

        this_frame.append(surface_tensor)
        del surface_tensor
    
    # assemble row tensors
    col = []
    for row_idx in range(square):
        row = []
        for col_idx in range(row_idx*square, (row_idx+1)*square):
            try:
                row.append(this_frame[col_idx])
            except:
                row.append(torch.zeros([3,resize[0],resize[1]]))    
        row_tensor = torch.cat(row,dim = 2)
        #print(row_tensor.shape)
        col.append(row_tensor)
        
    # assemble whole tensor
    whole_tensor = torch.cat(col,dim = 1).to(encDevice) /255.0
    del this_frame
    #print(whole_tensor.shape)
    
    
    #frame = whole_tensor.data.numpy().astype(np.uint8).transpose(1,2,0).astype(np.uint8)
    # cv2.imshow("",frame)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    
    surface_rgb = tensor_to_surface(whole_tensor,0)
    dst_surface = to_nv12.run(surface_rgb)
    
    success = nvEnc.EncodeSingleSurface(dst_surface, encFrame)
    framesSent += 1

    if(success):
        encByteArray = bytearray(encFrame)
        encFile.write(encByteArray)
        framesReceived += 1
            
    frame_counter += 1
    
    del whole_tensor,surface_rgb,dst_surface


#Encoder is asynchronous, so we need to flush it
# Encoder is asynchronous, so we need to flush it
while True:
        success = nvEnc.FlushSinglePacket(encFrame)
        if(success):
            byteArray = bytearray(encFrame)
            encFile.write(byteArray)
            framesReceived += 1
            framesFlushed += 1

        else:
            break


print(framesReceived, '/', frame_counter,
      ' frames encoded and written to output file.')
print(framesFlushed, ' frame(s) received during encoder flush.')

sys.exit(0)
            
            # This is optional and depends on what you NN expects to take as input
            # Normalize to range desired by NN. Originally it's 
            #surface_tensor = surface_tensor.type(dtype=torch.cuda.FloatTensor)/255.0
                 