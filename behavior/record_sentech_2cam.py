#########################
##### Pankaj Gupta#######
# Record behavior video from two sentech cameras at the same time.

import os
import sys
sys.path.append('..')
sys.path.append(os.getcwd())
from CameraFactory import CameraFactory
from imutils.video import FPS
import numpy as np
import math
import cv2
import imutils
import time
from datetime import datetime
from threading import *
import RPi.GPIO as GPIO
from Arduino import Arduino
import cvui
import tables
import roi_manager
from collections import deque
import re
import json
from configparser import ConfigParser
import csv
import helper as clh
# import board
# import busio
# import adafruit_mpr121
from audiostream import get_output
from audiostream.sources.wave import SineSource
import warnings
import random as rm
import itertools

warnings.filterwarnings("ignore", category=DeprecationWarning)

configPath = '../config.ini'
config      = ConfigParser()
config.read(configPath)
cfg = 'sentech_give_rewards'
cfgDict = dict(config.items(cfg))
vidSourceName = cfgDict['vid_source']
data_root   = cfgDict['data_root']
image_stream_filename = cfgDict['video_file']
res     = list(map(int, cfgDict['resolution'].split(', ')))
fr = int(cfgDict['framerate'])
audio = float(cfgDict['audio'])
total_trials = int(cfgDict['total_trials'])
maxTrialDur = int(cfgDict['max_trial_dur'])
startrest_duration = int(cfgDict['start_rest_dur'])
endrest_duration = int(cfgDict['end_rest_dur'])
n_tones = 0
audiodelay = 0
sessionType = clh.SessionType.behavior_recording

#list of sessions. session_name,reward_threshold_mm
sessions = [('S1',0)]

pinBrainTTL = 13
ledFailTrial = 12
ledReward = 7
ledLightTTL = 40
# board = Arduino("115200", port="/dev/ttyACM0") # plugged in via USB, serial com at rate 115200
# board.pinMode(pinBrainTTL, "OUTPUT")
# board.pinMode(ledReward, "OUTPUT")
# board.pinMode(ledFailTrial, "OUTPUT")
# board.pinMode(ledLightTTL, "OUTPUT")

GPIO.setmode (GPIO.BOARD)
GPIO.setup(pinBrainTTL,GPIO.OUT, initial=GPIO.LOW)
GPIO.setup(ledReward,GPIO.OUT, initial=GPIO.LOW)
GPIO.setup(ledFailTrial,GPIO.OUT, initial=GPIO.LOW)
GPIO.setup(ledLightTTL,GPIO.OUT, initial=GPIO.LOW)

ts_trigger = 0
ts_detection = 0
run_threads = True
runPreview = True
runRecording = True

WINDOW_NAME = 'Closed loop'

cvui.init(WINDOW_NAME)
cvui.init(WINDOW_NAME + '2')

#blink 'led' once for 't' length
def timed_led(led, t):
    global fr, run_threads
    GPIO.output(led, GPIO.HIGH)
    while run_threads:
        time.sleep(1/fr)
    GPIO.output(led, GPIO.LOW)

def blink_led(led, t):
    GPIO.output(led, GPIO.HIGH)
    time.sleep(t)
    GPIO.output(led, GPIO.LOW)
def blink_led_arduino(led, t):
    board.digitalWrite(led, "HIGH")
    # board.analogWrite(led, 180)
    time.sleep(t)
    board.digitalWrite(led, "LOW")
    # board.analogWrite(led, 0)

#continuously blink 'led' for 't' length and 'gap' between blinks
def cont_blink_led(led, t, gap):
    print("blink led continuous")
    global ts_trigger
    global run_threads
    while(run_threads):
        ts_trigger = time.time()
        blink_led(led,t)
        time.sleep(gap)
        
summary_filename    = data_root + os.sep + config.get(cfg, 'summary_file')
summary_exists      = os.path.isfile(summary_filename)
summaryfile         = open (summary_filename, 'a', encoding="utf-8")
headers             = [col.strip() for col in config.get(cfg, 'summary_header').split(',')]
writer              = csv.DictWriter(summaryfile, delimiter=',', lineterminator='\n',fieldnames=headers)
if not summary_exists:
    writer.writeheader()  # file doesn't exist yet, write a header

print("Start preview\n\n")
# initialize the camera and grab a reference to the raw camera capture
exec('from ' + vidSourceName + ' import ' + vidSourceName)
# initialize the camera and grab a reference to the raw camera capture
deviceIdx = 0
vs1 = CameraFactory(eval(vidSourceName+'(cfgDict, deviceIdx).start()'))
image1 = vs1.get_image()
deviceIdx = 1
vs2 = CameraFactory(eval(vidSourceName+'(cfgDict, deviceIdx).start()'))
image2 = vs2.get_image()

GPIO.output(ledLightTTL, GPIO.HIGH)
last_time=time.time()
while runPreview:
    start = time.time()
    #image = frame.array
    image1 = vs1.get_image()
    if image1 is None:
        print('Image1 none')
        continue
    image1 = image1[-res[1]:, -res[0]:, :]

    image2 = vs2.get_image()
    if image2 is None:
        print('Image2 none')
        continue
    image2 = image2[-res[1]:, -res[0]:, :]

    cvui.imshow(WINDOW_NAME, cv2.resize(image1,(image1.shape[1]*2,image1.shape[0]*2)))
    cvui.imshow(WINDOW_NAME + '2', cv2.resize(image2,(image2.shape[1]*2,image2.shape[0]*2)))
    
    # Press Esc or Ctrl-C to stop the program
    k = cv2.waitKey(1)
    if k == 27:
        runPreview = False
        break
    
    print('FPS', round(1 / (time.time() - last_time), 2), end='\r')
    last_time = time.time()
    # time.sleep(max(1./fr - (time.time() - start), 0))

#co save the configuration used in current directory
with open(configPath, 'w', encoding="utf-8") as f:
        config.write(f)

GPIO.output(ledLightTTL, GPIO.LOW)

mouse_id = input("Please enter mouse ID: ")
tr = 0
# Get the current time and initialize the project folder
tm = datetime.now()
data_name = str(tm.year) + \
            format(tm.month, '02d') + \
            format(tm.day, '02d') + \
            format(tm.hour, '02d') + \
            format(tm.minute, '02d') + \
            format(tm.second, '02d')
data_root = data_root + os.sep + mouse_id + os.sep + data_name

if not os.path.exists(data_root):
        print("Creating data directory: ",data_root)
        os.makedirs(data_root)

for session, reward_threshold_mm in sessions:
    
    if runRecording:
        #runningImgQueue.clear()
        #runningImgQueue.append(np.zeros((res[0], res[1], 3)))
        totRewards = 0

        # Get the current time and initialize the project folder
        tm = datetime.now()
        session_root = data_root
        if not os.path.exists(session_root):
            print("Creating data directory: ",session_root)
            os.makedirs(session_root)

        image1 = vs1.get_image()
        image1 = image1[-res[1]:, -res[0]:, :]
        rows, cols, channels = image1.shape

        image2 = vs2.get_image()
        image2 = image2[-res[1]:, -res[0]:, :]
        
        write_cfg = ConfigParser()
        write_cfg['configsection'] = {'config': cfg}
        write_cfg[cfg] = dict(config.items(cfg))
        write_cfg.set(cfg, 'reward_threshold_mm',str(reward_threshold_mm))
        # save the configuration used in target data directory
        with open(session_root + os.sep + mouse_id + '_' + data_name +'_config.ini', 'w', encoding="utf-8") as f:
            write_cfg.write(f)
        
        video_path = session_root + os.sep + mouse_id + '_' + data_name + '_1.mp4'
        fourcc = cv2.VideoWriter_fourcc(*'MP4V')
        video_writer1= cv2.VideoWriter(video_path, fourcc, fr, (640,320))

        video_path = session_root + os.sep + mouse_id + '_' + data_name + '_2.mp4'
        fourcc = cv2.VideoWriter_fourcc(*'MP4V')
        video_writer2= cv2.VideoWriter(video_path, fourcc, fr, (640,320))

        logFileName = session_root + os.sep + mouse_id + '_' + data_name + '.txt'
        logFile = open(logFileName, 'w', encoding="utf-8")
        logFile.write('frame' + '\t' +
                      'time' + '\t' +
                      'reward' + '\t' +
                      'trial' + '\t' +
                      'audio' + '\t' +
                      'lick' + '\n')
        print("Start recording\n")
    
    image1 = vs1.get_image()
    image1 = image1[-res[1]:, -res[0]:, :]

    #send TTL pulse for brain imaging to start recording
    # board.digitalWrite(pinBrainTTL, "HIGH")
    # board.digitalWrite(pinBrainTTL, "LOW")
    GPIO.output(ledLightTTL, GPIO.HIGH)
    #sessDurSec = total_trials * (failrest_duration + maxTrialDur)
    #lightTmr = Timer(1,timed_led, args=(ledLightTTL,sessDurSec))
    #lightTmr.start()
    fps = FPS().start()
    #import pdb; pdb.set_trace()
    restTimer = time.time()
    rest = True
    runTrial = False
    nTrial = 0
    rewardsInEpoch = 0
    rewardThreshTimer = time.time()
    lastRewTime = time.time()
    
    deliverReward = True
    rest_duration = endrest_duration

    while image1 is not None and nTrial <= total_trials:
        start = time.time()
        
        image1 = vs1.get_image()
        image1 = image1[-res[1]:, -res[0]:, :]
        image2 = vs2.get_image()
        image2 = image2[-res[1]:, -res[0]:, :]

        reward = 0
        touch = 0
        
        if image1 is None:
            break

        fps.update()
        
        video_writer1.write(image1)
        video_writer2.write(image2)

        sttime = datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')
        logFile.write(str(fps._numFrames) + '\t' +
                      sttime + '\t' +
                      str(reward) + '\t' +
                      str(tr) + '\t' +
                      str(audio) + '\t' +
                      str(touch) + '\n')
        
        cvui.imshow(WINDOW_NAME, cv2.resize(image1,(image1.shape[1],image1.shape[0])));
        cvui.imshow(WINDOW_NAME + '2', cv2.resize(image2,(image2.shape[1],image2.shape[0])));
        # Press Esc or Ctrl-C to stop the program
        k = cv2.waitKey(1)
        if k == 27:
            break
        
        time.sleep(max(1./fr - (time.time() - start), 0))
        #vs.camera_factory.updatesleep(max(0.5/(fr) - (time.time() - start), 0.0))
    
    GPIO.output(ledLightTTL, GPIO.LOW)
    
    #send TTL pulse for brain imaging to stop recording. LOW HIGH LOW sequence to deal with XCAP only recognizing rising edge.
    # board.digitalWrite(pinBrainTTL, "HIGH")
    # board.digitalWrite(pinBrainTTL, "LOW")
    # board.digitalWrite(pinBrainTTL, "HIGH")
    # board.digitalWrite(pinBrainTTL, "LOW")
    
    ############ Record for some time longer to see the LED's turn off #############
    t_end = time.time() + 2
    while time.time() < t_end:
        start = time.time()
        image1 = vs1.get_image()
        image1 = image1[-res[1]:, -res[0]:, :]
        image2 = vs2.get_image()
        image2 = image2[-res[1]:, -res[0]:, :]
        #imagecopy= np.copy(image1)

        fps.update()
        
        video_writer1.write(image1)
        video_writer2.write(image2)
        sttime = datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')
        logFile.write(str(fps._numFrames) + '\t' +
                      sttime + '\t' +
                      str(0) + '\t' +
                      str(0) + '\t' +
                      str(0) + '\t' +
                      str(0) + '\n')
        cvui.imshow(WINDOW_NAME, cv2.resize(image1,(image1.shape[1],image1.shape[0])))
        cvui.imshow(WINDOW_NAME +'2', cv2.resize(image2,(image2.shape[1],image2.shape[0])))
        k = cv2.waitKey(1)
        if k == 27:
            break
        time.sleep(max(1./fr - (time.time() - start), 0))
    ###############################################################################

    fps.stop()
    vs1.stop()
    vs2.stop()
    print('total rewards: ' + str(totRewards))
    video_writer1.release()
    video_writer2.release()
    logFile.close()

    writer.writerow({headers[0]: mouse_id,
                     headers[1]: session,
                     headers[2]: session_root,
                     headers[3]: fps._start,
                     headers[4]: fps._end,
                     headers[5]: fps.elapsed(),
                     headers[6]: fps.fps(),
                     headers[7]: 0,
                     headers[8]: 0,
                     headers[9]: 0,
                     headers[10]: 0,
                     headers[11]: 0,
                     headers[12]: sessionType.name})
    print("[INFO] elasped time: {:.2f}".format(fps.elapsed()))
    print("[INFO] approx. FPS: {:.2f}\n\n".format(fps.fps()))

run_threads = False

summaryfile.close()

cv2.destroyAllWindows()


