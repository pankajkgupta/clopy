# import the necessary packages
from picamera.array import PiRGBArray
from picamera import PiCamera
from threading import Thread
import cv2
import time

class PiCameraStream:
    def __init__(self, cfgDict=None):
        # initialize the camera and stream
        self.camera = PiCamera()
        if 'resolution' in cfgDict.keys():
            self.camera.resolution = list(map(int, cfgDict['resolution'].split(', ')))
        if 'framerate' in cfgDict.keys():
            self.camera.framerate = int(cfgDict['framerate'])
        if 'awb_mode' in cfgDict.keys():
            self.camera.awb_mode = cfgDict['awb_mode']
        self.camera.awb_gains = (1,1)
        if 'shutter_speed' in cfgDict.keys():
            self.camera.shutter_speed = int(cfgDict['shutter_speed'])
        if 'dff_history' in cfgDict:
            while True:
                if(self.camera.analog_gain>=4.0 and
                   self.camera.analog_gain<=5.0 and
                   self.camera.digital_gain>=1 and
                   self.camera.digital_gain<=1.5):
                    self.camera.exposure_mode='off'
                    print("ok")
                    break
                else:
                    print('analog gain: ' + str(eval(str(self.camera.analog_gain))))
                    print('digital gain: ' + str(eval(str(self.camera.digital_gain))))
        if 'iso' in cfgDict.keys():
            self.camera.iso = int(cfgDict['iso'])
        
        if self.camera.revision == 'ov5647': #camera module v1
            if 'sensor_mode_v1' in cfgDict.keys():
                self.camera.sensor_mode = int(cfgDict['sensor_mode_v1'])
            else:
                if 'sensor_mode' in cfgDict.keys():
                    self.camera.sensor_mode = int(cfgDict['sensor_mode'])
        else: #camera module v2
            self.camera.sensor_mode = int(cfgDict['sensor_mode'])
        #self.camera.exposure_mode = 'off' 'fixedfps'
        self.camera.exposure_mode = 'off'
        self.rawCapture = PiRGBArray(self.camera, size=self.camera.resolution)
        self.stream = self.camera.capture_continuous(self.rawCapture,
            format="bgr", use_video_port=True)

        # initialize the frame and the variable used to indicate
        # if the thread should be stopped
        self.frame = None
        self.stopped = False
        self.save_path = ""
        self.updateSleep = 0.0
        print('analog gain: ' + str(self.camera.analog_gain))

    def start(self):
        # start the thread to read frames from the video stream
        t = Thread(target=self.update, args=())
        t.daemon = True
        t.start()
        return self

    def update(self):
        # keep looping infinitely until the thread is stopped
        #self.camera.start_recording("test.h264", format='rgb')
        if self.save_path:
                    self.camera.start_recording(self.save_path, format='rgb')
        for f in self.stream:
            start = time.time()
            # grab the frame from the stream and clear the stream in
            # preparation for the next frame
            self.frame = f.array
            self.rawCapture.seek(0)

            # if the thread indicator variable is set, stop the thread
            # and resource camera resources
            if self.stopped:
                self.stream.close()
                if self.save_path:
                    self.camera.stop_recording()
                self.rawCapture.close()
                self.camera.close()
                return
            time.sleep(max(0.5/(self.camera.framerate) - (time.time() - start), 0.0))
        
    def read(self):
        # the frame most recently read
        return self.frame
        #import pdb; pdb.set_trace()
#        im= next(self.stream).array
#        self.rawCapture.truncate()
#        self.rawCapture.seek(0)
#        return im
    def clearimg(self):
            self.frame = None
            return

    def stop(self):
        # indicate that the thread should be stopped
        self.stopped = True
        
#        self.stream.close()
#        if self.save_path:
#            self.camera.stop_recording()
#        self.rawCapture.close()
#        self.camera.close()
