# import the necessary packages
from threading import Thread
import cv2
import time
import tables
import os

class VideoStream:
    def __init__(self, cfgDict=None):
        # initialize the frame and the variable used to indicate
        # if the thread should be stopped
        self.frame = None
        self.stopped = False
        self.savePath = None
        self.curFrame = 0

        # open the hdf5 file
        self.image_hdf5_file = tables.open_file(cfgDict['vid_root'] + os.sep + cfgDict['raw_image_file'], mode='r')
        # Total number of samples
        self.image_num = self.image_hdf5_file.root.raw_images.shape[0]


    def start(self):
        return self

    def update(self):
        return self

    def read(self):
        if self.curFrame >= self.image_num:
            self.frame = None
        else:
            self.frame = self.image_hdf5_file.root.raw_images[self.curFrame]
            self.curFrame += 1
        return self.frame

    def clearimg(self):
        self.frame = None
        return

    def stop(self):
        # indicate that the thread should be stopped
        self.stopped = True
        self.image_hdf5_file.close()
