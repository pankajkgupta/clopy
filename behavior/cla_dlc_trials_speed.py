#########################
##### Pankaj Gupta#######
# Closed loop audio feedback of DLC tracking. It is trial based.
# Reward is generated if target reached within specified time. 
# If target is not reached, a negative feedback in the form of buzzer is provided

### Habituation ####
# Trial starts if tracked paw in roi
# Success if paw moves to target roi within trial duration, else a fail (no-reward)
# 80% successful trials are randomly rewarded

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
import ClosedLoopHelper2ROI as clh
import roi_manager
from collections import deque
import re
import json
from configparser import ConfigParser
import csv
# import board
# import busio
# import adafruit_mpr121
from audiostream import get_output
from audiostream.sources.wave import SineSource
import warnings
import random as rm
from dlclive import DLCLive, Processor
import itertools

warnings.filterwarnings("ignore", category=DeprecationWarning)

configPath = '../config.ini'
config      = ConfigParser()
config.read(configPath)
cfg = 'sentech_dlclive'
cfgDict = dict(config.items(cfg))
vidSourceName = cfgDict['vid_source']
data_root   = cfgDict['data_root']
image_stream_filename = cfgDict['video_file']
res     = list(map(int, cfgDict['resolution'].split(', ')))
fr = int(cfgDict['framerate'])
iso = int(cfgDict['iso'])
#dff_hostory is in seconds. Multiply with framerate to get array length
len_joint_history = int(cfgDict['joint_history_sec']) * fr
anchor = cvui.Point()
control_point = cfgDict['control_point']
start_center = list(map(int, cfgDict['start_center'].split(',')))
target_center = list(map(int, cfgDict['target_center'].split(',')))
ppmm = float(cfgDict['ppmm'])
# roi_size in config file is in mm. We convert it to pixel coordinates and round off.
start_radius = int(float(cfgDict['start_radius_mm']) * ppmm)
target_radius = int(float(cfgDict['target_radius_mm']) * ppmm)
# roi_size = [int(round(x)) for x in np.array([float(i) for i in cfgDict['roi_size_mm'].split(',')]) * ppmm]
total_trials = int(cfgDict['total_trials'])
maxTrialDur = int(cfgDict['max_trial_dur'])
rest_duration = int(cfgDict['initial_rest_dur'])
successrest_duration = int(cfgDict['success_rest_dur'])
failrest_duration = int(cfgDict['fail_rest_dur'])
bregma = list(map(int, cfgDict['bregma'].split(', ')))
dlc_model_path = cfgDict['dlc_model_path']
audio_tr_prob = float(cfgDict['audio'])
n_tones = int(cfgDict['n_tones'])
audiodelay = int(cfgDict['audio_delay'])
freqQue = deque(maxlen=(audiodelay*fr)+1)
freqQue.extend([1000]*(audiodelay*fr))
rewarddelay = int(cfgDict['reward_delay'])
reward_threshold_mm = float(cfgDict['reward_threshold_mm'])
adaptive_threshold = int(cfgDict['adaptive_threshold'])
speed_threshold = int(cfgDict['speed_threshold'])
# sessionType = clh.SessionType.single_audio_normal_reward
sessionType = clh.SessionType.normal_audio_normal_reward
# sessionType = clh.SessionType.no_audio_random_reward
# sessionType = clh.SessionType.no_audio_no_reward
# sessionType = clh.SessionType.normal_audio_no_reward

start_roi_manager = roi_manager.Circle('start_center', x=start_center[0], y= start_center[1],
                                     r= start_radius, color=[0, 200, 0])
target_roi_manager = roi_manager.Circle('target_center', x=target_center[0], y= target_center[1],
                                      r= target_radius, color=[0, 0, 200])
rois = []
rois.append(start_roi_manager)
rois.append(target_roi_manager)

# # Create I2C bus.
# i2c = busio.I2C(board.SCL, board.SDA)

# # Create MPR121 object.
# touchCap = adafruit_mpr121.MPR121(i2c)

#list of sessions. session_name,reward_threshold_mm
sessions = [('S1',reward_threshold_mm)]

working = False

n_aud_ch = 1
au_rate = 44100
audio_tr_arr = rm.sample([1]*int(audio_tr_prob*total_trials) + [0]*int((1-audio_tr_prob)*total_trials),k=total_trials)
# insert audio=0 for 0th trial
#audio_tr_arr = np.insert(audio_tr_arr, 0, 0)
#freqDict = clh.get_freqs(n_tones)
#get frequencies in range 1-24k with quarter octave increment
freqs = [1000 * (2**(1/4))**i for i in range(n_tones)]
#freqs = np.interp(np.arange(0, len(freqs), 0.5), np.arange(0, len(freqs)), freqs).astype(int)

speed_bins = np.linspace(0, speed_threshold, len(freqs)-1) # -1 because we are going to insert inf below
## set upper limit to inf so speed indexing never goes out of range
speed_bins = np.append(speed_bins, np.inf)
cj_speed_reward_que = deque(maxlen=(rewarddelay*fr)+1)
cj_speed_reward_que.extend([0]*(rewarddelay*fr))
cj_speed_hist_que = deque(maxlen=len_joint_history+1)
cj_speed_hist_que.extend([np.inf]*(len_joint_history))

pinBrainTTL = 13
ledFailTrial = 12
ledReward = 7
ledLightTTL = 40
board = Arduino("115200", port="/dev/ttyACM0") # plugged in via USB, serial com at rate 115200
board.pinMode(pinBrainTTL, "OUTPUT")
board.pinMode(ledReward, "OUTPUT")
board.pinMode(ledFailTrial, "OUTPUT")
board.pinMode(ledLightTTL, "OUTPUT")

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
roi_manager.init(roi_manager.annots, rois, WINDOW_NAME, res[0], res[1])
cv2.setMouseCallback(WINDOW_NAME, roi_manager.dragcircle, roi_manager.annots)

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
vs = CameraFactory(eval(vidSourceName+'(cfgDict).start()'))
image = vs.get_image()
dlc_proc            = Processor()
dlc_live            = DLCLive(dlc_model_path, processor=dlc_proc, display=False)
all_joints          = dlc_live.cfg['all_joints']
all_joints_names    = dlc_live.cfg['all_joints_names']
control_point_ix    = all_joints_names.index(control_point)
dlc_live.init_inference(image)

GPIO.output(ledLightTTL, GPIO.HIGH)
last_time=time.time()
while runPreview:
    start = time.time()
    #image = frame.array
    image = vs.get_image()
    if image is None:
        print('Image none')
        continue
    # breakpoint()
    image = image[-res[1]:, -res[0]:, :]
    # if touchCap[0].value:
    #      print('Pin 0 touched!')
#     print('analog gain: ' + str(eval(str(vs.camera_factory.camera.analog_gain))))
#     print('digital gain: ' + str(eval(str(vs.camera_factory.camera.digital_gain))))
#     print('ISO: ' + str(eval(str(vs.camera_factory.camera.iso))))
    
    pose = dlc_live.get_pose(image)
    for point in pose:
        cv2.circle(image, (int(point[0]), int(point[1])), radius=2, color=(0, 0, 255), thickness=-3)
    # # Render the roi
    # for roi in rois:
    #     cv2.circle(image, (roi.x, roi.y), radius=roi.r, color=roi.color, thickness=2)
    cv2.rectangle(image, (355, 15), (360, 20), color=(255, 0, 255), thickness=1) # align this rectangle to left of the visible headfixing post
    # This function must be called *AFTER* all UI components. It does
    # all the behind the scenes magic to handle mouse clicks, etc.
    # cvui.update()
    # cvui.imshow(WINDOW_NAME, image)
    cvui.imshow(WINDOW_NAME, cv2.resize(image,(image.shape[1]*2,image.shape[0]*2)))
    
    # Press Esc or Ctrl-C to stop the program
    k = cv2.waitKey(1)
    if k == 27:
        runPreview = False
        break
    
    print('FPS', round(1 / (time.time() - last_time), 2), end='\r')
    last_time = time.time()
    # time.sleep(max(1./fr - (time.time() - start), 0))

for roi in rois:
    config.set(cfg, roi.name,str(roi.x)+','+str(roi.y))
#co save the configuration used in current directory
with open(configPath, 'w', encoding="utf-8") as f:
        config.write(f)

GPIO.output(ledLightTTL, GPIO.LOW)

mouse_id = input("Please enter mouse ID: ")
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

        image = vs.get_image()
        image = image[-res[1]:, -res[0]:, :]
        rows, cols, channels = image.shape
        
        write_cfg = ConfigParser()
        write_cfg['configsection'] = {'config': cfg}
        write_cfg[cfg] = dict(config.items(cfg))
        write_cfg.set(cfg, 'reward_threshold_mm',str(reward_threshold_mm))
        # save the configuration used in target data directory
        with open(session_root + os.sep + mouse_id + '_' + data_name +'_config.ini', 'w', encoding="utf-8") as f:
            write_cfg.write(f)
        
        video_path = session_root + os.sep + mouse_id + '_' + data_name + '.mp4'
        fourcc = cv2.VideoWriter_fourcc(*'MP4V')
        video_writer= cv2.VideoWriter(video_path, fourcc, fr, (image.shape[1],image.shape[0]))

        pose_columns = ["_".join(x) for x in itertools.product(all_joints_names,['x','y', 'likelihood'])]
        logFileName = session_root + os.sep + mouse_id + '_' + data_name + '.txt'
        logFile = open(logFileName, 'w', encoding="utf-8")
        logFile.write('frame' + '\t' +
                      'time' + '\t' +
                      'cj_speed' + '\t'
                      'freq' + '\t' +
                      'reward' + '\t' +
                      'trial' + '\t' +
                      'audio' + '\t' +
                      'lick' + '\t' +
                      '\t'.join(pose_columns) + '\n')
        print("Start recording\n")
    
    image = vs.get_image()
    image = image[-res[1]:, -res[0]:, :]
    pose = dlc_live.get_pose(image)
    cj_prev = pose[control_point_ix][0:2]

    #send TTL pulse for brain imaging to start recording
    board.digitalWrite(pinBrainTTL, "HIGH")
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
    
    ####################
    # get a output stream where we can play samples
    austream = get_output(channels=n_aud_ch, rate=au_rate, buffersize=128)
    # create one wave sin() at 220Hz, attach it to our speaker, and play
    sinsource = SineSource(austream, 1000)
    sinsource.stop()
    #############
    
    while image is not None and nTrial <= total_trials:
        start = time.time()
        
        image = vs.get_image()
        image = image[-res[1]:, -res[0]:, :]
        imagecopy= np.copy(image)
        #print('AWB gain: ' + str(vs.camera_factory.camera.awb_gains))
        reward = 0
        touch = 0
        
        # if touchCap[0].value:
        #      print('Lick!')
        #      touch = 1
        
        if image is None:
            break

        fps.update()
        
        pose = dlc_live.get_pose(image)
        # for roi, imQ in zip(rois,runningImgQuROI):
        #     imROI = image[roi.y : roi.y + roi.h, roi.x : roi.x + roi.w,:]
            
        #     runAvg = np.mean(imQ, axis=0)
        #     dffROI = imROI - runAvg
        #     dffROI = np.divide(dffROI, runAvg, out=np.zeros_like(dffROI), where=runAvg!=0)
        #     dffGreen = dffROI[:,:,1]
        #     dffBlue = dffROI[:,:,0]
        #     dffCorrected = dffGreen - dffBlue
        #     imQ.append(imROI)
        #     roi.avgDff = np.sum(dffCorrected)/roi.area()
        for point in pose:
            cv2.circle(image, (int(point[0]), int(point[1])), radius=2, color=(0, 0, 255), thickness=-3)
        # # Render the roi
        # for roi in rois:
        #     cv2.circle(image, (roi.x, roi.y), radius=roi.r, color=roi.color, thickness=2)
            
        cj_speed = math.dist(cj_prev, pose[control_point_ix][0:2])
        # cj_target_dist = math.dist(target_center, pose[control_point_ix][0:2])
        # cj_start_dist = math.dist(start_center, pose[control_point_ix][0:2])
        
        cj_speed_reward_que.append(cj_speed)
        cj_speed_delayed = cj_speed_reward_que.popleft()

        cj_speed_hist_que.append(cj_speed)
        
        fq = int(freqs[np.searchsorted(speed_bins, cj_speed)])
        freqQue.append(fq)
        freq = freqQue.popleft()
        if runTrial:
            print('Trial time: ', round(time.time() - trialTimer, 2), end='\r')
            tr = nTrial
            rest_duration = failrest_duration

            ## NORMAL REWARDS
            if sessionType == clh.SessionType.normal_audio_normal_reward:
                deliverReward = cj_speed_delayed > speed_threshold # joint speed is greater than threshold
            if sessionType == clh.SessionType.single_audio_normal_reward:
                freq = 1000
                deliverReward = cj_speed_delayed > speed_threshold # joint speed is greater than threshold
            ## RANDOM REWARDS in trial periods
            if sessionType == clh.SessionType.no_audio_random_reward:
                deliverReward = np.random.choice(np.arange(0, 2), p=[0.998, 0.002])
            if sessionType == clh.SessionType.normal_audio_no_reward:
                deliverReward = cj_speed_delayed > speed_threshold
            if sessionType == clh.SessionType.no_audio_no_reward:
                deliverReward = cj_speed_delayed > speed_threshold
            if deliverReward:
                if not (sessionType == clh.SessionType.normal_audio_no_reward or sessionType == clh.SessionType.no_audio_no_reward):
                    reward = 1
                    totRewards += 1
                    rewardsInEpoch += 1
                    t2 = Thread(target=blink_led_arduino, args=(ledReward, 0.1))
                    t2.start()
                    lastRewTime = time.time()
                #End the trial since reward obtained
                #rest = True
                #runTrial = False
                print("control joint speed is: \t", cj_speed_delayed)
                print("speed threshold is: \t", speed_threshold)
                print('total rewards: ' + str(totRewards), end="\n")

                runTrial = False
                rest_duration = successrest_duration
            
            elif time.time() - trialTimer >= maxTrialDur:
                runTrial = False
                reward = -1
                t2 = Thread(target=blink_led_arduino, args=(ledFailTrial, 0.5))
                t2.start()
            restTimer = time.time()
            #play_tone(freq, 0.1, stream)
            sinsource.frequency = freq
            #print("Freq is: ",freq)
        else:
            if not rest: #trial just ended
                sinsource.stop()
                print('\n########################### End #################################')
            rest = True
            runTrial = False

            if rest and time.time() - restTimer < rest_duration: # rest for the rest_duration
                print('Rest time: ', round(time.time() - restTimer, 2), end='\r')
                freq = 0
                tr = 0
                audio = 0

            else:
                print('Rest timer out', round(time.time() - restTimer, 2), end = '\t')

                ############# Block for trials to only start when paw is stable for specified duration ###############
                if np.mean(cj_speed_hist_que) < 1: # only if control joint is not moving for specified duration, start the next trial
                    print('\nPaw is stable')
                    print("\n######################### Trial: ",nTrial+1, " ###############################")
                    rest = False
                    runTrial = True
                    trialTimer = time.time()
                    audio = audio_tr_arr[tr]
                    if audio:
                        austream = get_output(channels=n_aud_ch, rate=au_rate, buffersize=128)
                        sinsource = SineSource(austream, freq)
                        sinsource.start()
                else:
                    print('Wait for paw to be stable', end='\r')
                ##########################################################
                ############ Uncomment this block for trials  to start as soon as rest timer finishes ####################
                # print("\n######################### Trial: ",nTrial+1, " ###############################")
                # rest = False
                # runTrial = True
                # trialTimer = time.time()
                # audio = audio_tr_arr[tr]
                # if audio:
                #     austream = get_output(channels=n_aud_ch, rate=au_rate, buffersize=128)
                #     sinsource = SineSource(austream, freq)
                #     sinsource.start()
                ##########################################################
                ############ OR uncomment below block if the trials should only start when joint is in start ROI #############
                # if cj_start_dist < start_radius: # only if control joint is within start-roi, start the next trial
                #     print('Paw in the start ROI')
                #     print("\n######################### Trial: ",nTrial+1, " ###############################")
                #     rest = False
                #     runTrial = True
                #     trialTimer = time.time()
                #     audio = audio_tr_arr[tr]
                #     if audio:
                #         austream = get_output(channels=n_aud_ch, rate=au_rate, buffersize=128)
                #         sinsource = SineSource(austream, freq)
                #         sinsource.start()
                # else:
                #     print('Paw not in start ROI', end='\r')
                ##########################################################
    
        # check if adaptive threshold is enabled every 15 seconds
        if adaptive_threshold and time.time() - rewardThreshTimer > 30:
            # if more than 1 reward dispensed in last epoch, increase the threshold
            if rewardsInEpoch > 1:
                speed_threshold += 2
            # if no reward was dispensed in last epoch, decrease the threshold
            if rewardsInEpoch ==0:
                speed_threshold = max(speed_threshold-2, 2)

            rewardsInEpoch = 0
            rewardThreshTimer = time.time()
            # regenerate the dist bins to frequency mappings
            speed_bins = np.linspace(0, speed_threshold, len(freqs)-1) # -1 because we are going to insert inf below
            ## set upper limit to inf so distance indexing never goes out of range
            speed_bins = np.append(speed_bins, np.inf)
        
        if tr ==0 and runTrial:
            nTrial += 1
        
        video_writer.write(imagecopy)
        # image_storage.append(image[None])
        cj_prev = pose[control_point_ix][0:2]

        sttime = datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')
        logFile.write(str(fps._numFrames) + '\t' +
                      sttime + '\t' +
                      str(cj_speed) + '\t' +
                      str(freq) + '\t' +
                      str(reward) + '\t' +
                      str(tr) + '\t' +
                      str(audio) + '\t' +
                      str(touch) + '\t' +
                      '\t'.join([str(x) for x in pose.flatten()]) + '\n')
        
        cvui.imshow(WINDOW_NAME, cv2.resize(image,(image.shape[1]*2,image.shape[0]*2)));
        # Press Esc or Ctrl-C to stop the program
        k = cv2.waitKey(1)
        if k == 27:
            break
        
        time.sleep(max(1./fr - (time.time() - start), 0))
        #vs.camera_factory.updatesleep(max(0.5/(fr) - (time.time() - start), 0.0))
    
    GPIO.output(ledLightTTL, GPIO.LOW)
    
    #send TTL pulse for brain imaging to stop recording. LOW HIGH LOW sequence to deal with XCAP only recognizing rising edge.
    # board.digitalWrite(pinBrainTTL, "HIGH")
    board.digitalWrite(pinBrainTTL, "LOW")
    board.digitalWrite(pinBrainTTL, "HIGH")
    board.digitalWrite(pinBrainTTL, "LOW")
    
    ############ Record for some time longer to see the LED's turn off #############
    t_end = time.time() + 2
    while time.time() < t_end:
        start = time.time()
        image = vs.get_image()
        image = image[-res[1]:, -res[0]:, :]
        imagecopy= np.copy(image)
        pose = dlc_live.get_pose(image)
        fps.update()
        cj_speed = math.dist(cj_prev, pose[control_point_ix][0:2])
        video_writer.write(imagecopy)
        sttime = datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')
        logFile.write(str(fps._numFrames) + '\t' +
                      sttime + '\t' +
                      str(cj_speed) + '\t' +
                      str(0) + '\t' +
                      str(0) + '\t' +
                      str(0) + '\t' +
                      str(0) + '\t' +
                      str(0) + '\t' +
                      '\t'.join([str(x) for x in pose.flatten()]) + '\n')
        cvui.imshow(WINDOW_NAME, cv2.resize(image,(image.shape[1]*2,image.shape[0]*2)));
        k = cv2.waitKey(1)
        if k == 27:
            break
        time.sleep(max(1./fr - (time.time() - start), 0))
    ###############################################################################

    fps.stop()
    vs.stop()
    print('total rewards: ' + str(totRewards))
    video_writer.release()
    logFile.close()

    writer.writerow({headers[0]: mouse_id,
                     headers[1]: session,
                     headers[2]: session_root,
                     headers[3]: fps._start,
                     headers[4]: fps._end,
                     headers[5]: fps.elapsed(),
                     headers[6]: fps.fps(),
                     headers[7]: n_tones,
                     headers[8]: start_radius,
                     headers[9]: target_roi_manager.r,
                     headers[10]: totRewards,
                     headers[11]: audio_tr_prob,
                     headers[12]: sessionType.name})
    print("[INFO] elasped time: {:.2f}".format(fps.elapsed()))
    print("[INFO] approx. FPS: {:.2f}\n\n".format(fps.fps()))

run_threads = False

summaryfile.close()
#audPin.set_enable (0,1)
#stream.close()
sinsource.stop()
#p.terminate()
cv2.destroyAllWindows()


