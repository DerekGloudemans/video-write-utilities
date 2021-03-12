#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 11 21:29:21 2021

@author: worklab
"""


# using openCV, draw ignored regions for each frame. 
# Press d to remove all ignored regions
# otherwise they carry over from frame to frame
# Press 2 to advance frames, 1 to reverse frames
import numpy as np
import os
import cv2
import csv
import torch
import argparse
import _pickle as pickle
import cv2 as cv
from PIL import Image
from torchvision.transforms import functional as F

def plot_outputs(output_file,im_dir,ignore = None):
    
    if ignore is not None:
        with open(ignore,"rb") as f:
            ignored = pickle.load(f)
    else:
        ignored = None        
    
    frame_dict = {}
    
    with open(output_file) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            frame,obj_idx,left,top,width,height,conf,cls,visibility,_ = row
            frame = int(frame)
            obj_idx = int(obj_idx)
            left = float(left)
            top = float(top)
            width = float(width)
            height = float(height)
            conf = float(conf)
            cls = 0#torch.tensor(int(cls)).int()
            visibility = float(visibility)
            bbox = torch.tensor([left,top,left+width,top+height]).int()
            
            if frame not in frame_dict.keys():
                frame_dict[frame] = []
            
            frame_dict[frame].append([bbox,cls,conf,visibility,obj_idx])

    ims = [os.path.join(im_dir,item) for item in os.listdir(im_dir)]
    ims.sort()
    idx_colors = np.random.rand(10000,3)
    idx_colors[0] = np.array([0,1,0])
    idx_colors[1] = np.array([0,0,1])
    
    ff = 0
    for im_name in ims:
        
        frame = int(im_name.split("/")[-1].split(".jpg")[0])
        # open image
        
        im = Image.open(im_name)
        im = F.to_tensor(im)
        cv_im = np.array(im) 
        cv_im = np.clip(cv_im, 0, 1)
        
        # Convert RGB to BGR 
        cv_im = cv_im[::-1, :, :]         
        
        cv_im = np.moveaxis(cv_im,[0,1,2],[2,0,1])

        cv_im = cv_im.copy()

        
        im_idx = int(im_name.split("/")[-1].split(".jpg")[0])
        label = frame_dict[im_idx]
        
        for obj in label:
            bbox = obj[0]
            bbox = bbox.int().data.numpy()
            cls = obj[1]
            
            # check whether
            if ignored is not None:
                for region in ignored[frame]:
                    area = (bbox[2] - bbox[0])     * (bbox[3] - bbox[1])
                    xmin = max(region[0],bbox[0])
                    xmax = min(region[2],bbox[2])
                    ymin = max(region[1],bbox[1])
                    ymax = min(region[3],bbox[3])
                    intersection = max(0,(xmax - xmin)) * max(0,(ymax - ymin))
                    overlap = intersection  / (area)
                    
                    if overlap > 0.5:
                        cls = 1
            
            cv2.rectangle(cv_im,(bbox[0],bbox[1]),(bbox[2],bbox[3]), idx_colors[cls], 1)
            #plot_text(cv_im,(bbox[0],bbox[1]),obj_idx,0,class_colors,self.class_dict)
        
        
        # for region in metadata["ignored_regions"]:
        #     bbox = region.astype(int)
        #     cv2.rectangle(cv_im,(bbox[0],bbox[1]),(bbox[2],bbox[3]), class_colors[-1], 1)
       
    
        cv2.imshow("Frame",cv_im)
        cv2.waitKey(ff) 
        if ff == 0:
            ff = 1
    cv2.destroyAllWindows()


class Frame_Labeler():
    
    def __init__(self,directory,classes = ["Class 1","Class 2"]):
        self.frame = 1
        
        self.directory = directory
        self.frames = [os.path.join(directory,item) for item in os.listdir(directory)]
        self.frames.sort()
        
        
        self.frame_boxes = {}
        self.cur_frame_boxes = []
        self.cur_image = cv2.imread(self.frames[0])
        
        
        self.start_point = None # used to store click temporarily
        self.clicked = False 
        self.new = None # used to store a new box to be plotted temporarily
        self.cont = True


        # classes
        self.cur_class = 0
        self.n_classes = len(classes)
        self.class_names = classes
        self.colors = (np.random.rand(self.n_classes,3))*255
        self.colors[0] = np.array([0,0,255])

        
    def toggle_class(self):
        self.cur_class = (self.cur_class + 1) % self.n_classes
        print("Active Class: {}".format(self.class_names[self.cur_class]))
        
    def on_mouse(self,event, x, y, flags, params):
       
    
       if event == cv.EVENT_LBUTTONDOWN and not self.clicked:
         print("CLICK!")
         self.start_point = (x,y)
         self.clicked = True
         
       elif event == cv.EVENT_LBUTTONUP:
          print("CLICK RELASE")
          box = np.array([self.start_point[0],self.start_point[1],x,y,self.cur_class]).astype(int)
          self.cur_frame_boxes.append(box)
          self.new = box
          self.clicked = False
          
          
    
    def next(self):
        # store current boxes
        self.frame_boxes[self.frame] = self.cur_frame_boxes
        
        if self.frame == len(self.frames):
            print("Last Frame.")    
            
        else:
            self.frame += 1
            
            if self.frame in self.frame_boxes.keys():
                self.cur_frame_boxes = self.frame_boxes[self.frame] # keep boxes if already defined
                if len(self.frame_boxes[self.frame]) == 1:
                    self.cur_frame_boxes = self.frame_boxes[self.frame-1] # start with previous frame's boxes
            else:
                self.cur_frame_boxes = self.frame_boxes[self.frame-1] # start with previous frame's boxes
            
            # load image and plot existing boxes
            self.cur_image = cv2.imread(self.frames[self.frame-1])
            for box in self.cur_frame_boxes:
                self.cur_image = cv2.rectangle(self.cur_image,(box[0],box[1]),(box[2],box[3]),self.colors[box[4]],2)
    
    def prev(self):
        if self.frame == 1:
            print("On first frame. Cannot go to previous frame")
        else:
            self.frame -= 1
            self.cur_frame_boxes = self.frame_boxes[self.frame]
            
            # load image and plot existing boxes
            self.cur_image = cv2.imread(self.frames[self.frame-1])
            for box in self.cur_frame_boxes:
                self.cur_image = cv2.rectangle(self.cur_image,(box[0],box[1]),(box[2],box[3]),self.colors[box[4]],2)
          
            
            
    def quit(self):
        self.frame_boxes[self.frame] = self.cur_frame_boxes
        
        cv2.destroyAllWindows()
        self.cont = False
        print("Images are from {}".format(self.directory))
        name = input("Save file name (Enter q to discard):")
        if name == "q":
            print("Labels discarded")
        else:
            with open(name,"wb") as f:
                pickle.dump(self.frame_boxes,f)
            print("Saved boxes as file {}".format(name))
        
        
    def clear(self):
        self.cur_frame_boxes = []
        self.cur_image = cv2.imread(self.frames[self.frame-1])

    def undo(self):
        self.cur_frame_boxes = self.cur_frame_boxes[:-1]
        self.cur_image = cv2.imread(self.frames[self.frame-1])
        for box in self.cur_frame_boxes:
                self.cur_image = cv2.rectangle(self.cur_image,(box[0],box[1]),(box[2],box[3]),self.colors[box[4]],2)
                
                
    def run(self):  
        self.frame_boxes[self.frame] = self.cur_frame_boxes

        cv2.namedWindow("window")
        cv.setMouseCallback("window", self.on_mouse, 0)
           
        while(self.cont): # one frame
        
           
           
           if self.new is not None:
               self.cur_image = cv2.rectangle(self.cur_image,(self.new[0],self.new[1]),(self.new[2],self.new[3]),self.colors[self.new[4]],2)
               self.new = None
               
           cv2.imshow("window", self.cur_image)
           cv2.setWindowTitle("window",str(self.frame))
           
           key = cv2.waitKey(1)
           if key == ord('2'):
                self.next()
           elif key == ord('1'):
                self.prev()
           elif key == ord('c'):
                self.clear()
           elif key == ord("q"):
                self.quit()
           elif key == ord("0"):
                self.toggle_class()
           elif key == ord("3"):
               self.undo()
            
if __name__ == "__main__":
      
     #add argparse block here so we can optinally run from command line
     #add argparse block here so we can optinally run from command line
     try:
        parser = argparse.ArgumentParser()
        parser.add_argument("directory",help = "directory where frame images are stored")
        parser.add_argument("-classes", help = "list separated by commas",default = "Class1,Class2")


        args = parser.parse_args()
        
        dir = args.directory
        classes = args.classes
        
        frame_labeler = Frame_Labeler(dir,classes)
        frame_labeler.run()

    except:
        label_dir = "/home/worklab/Documents/derek/MOT-lbt/final_outputs/Submission 3 MOT20"               
        dataset = "/home/worklab/Data/cv/MOT20/test"
        for idx,track in enumerate(os.listdir(dataset)):
            if idx in [0,1]:
                continue
            
            dir = os.path.join(dataset,track,"img1")
            test = Frame_Labeler(dir)
            test.run()
            
            plot_outputs(os.path.join(label_dir,track+".txt"),dir,ignore = track+"_ignore.cpkl")
    