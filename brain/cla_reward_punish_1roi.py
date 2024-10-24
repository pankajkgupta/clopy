#########################
##### Pankaj Gupta#######
# Closed loop audio feedback script. It is trial based.
# Reward is generated if target reached within specified time. 
# If target is not reached, a negative feedback in the form of buzzer is provided

#from picamera.array import PiRGBArray
#from picamera import PiCamera
#following class is customised by me to set some camera parameters.
#It is originally part of imutils
import sys
sys.path.append('..')
from CameraFactory import CameraFactory
from imutils.video import FPS
import numpy as np
import cv2
import imutils
import time
from datetime import datetime
from threading import *
from gpiozero import LED
import cvui
import os
import tables
import helper as clh
import roi_manager
from collections import deque
import re
import json
from configparser import ConfigParser
import csv
import board
import busio
import adafruit_mpr121
from audiostream import get_output
from audiostream.sources.wave import SineSource
import warnings
import random as rm

warnings.filterwarnings("ignore", category=DeprecationWarning)

configPath = '../config.ini'
config      = ConfigParser()
config.read(configPath)
cfg = 'raspicambrain_reward_punish_1roi'
cfgDict = dict(config.items(cfg))
vidSourceName = cfgDict['vid_source']
data_root   = cfgDict['data_root']
image_stream_filename = cfgDict['raw_image_file']
res     = list(map(int, cfgDict['resolution'].split(', ')))
fr = int(cfgDict['framerate'])
iso = int(cfgDict['iso'])
sensormode = int(cfgDict['sensor_mode'])
#dff_hostory is in seconds. Multiply with framerate to get array length
dffHistory = int(cfgDict['dff_history']) * fr
anchor = cvui.Point()
roi_names = re.split('[-+/%]',cfgDict['roi'])
rois = []
ppmm = float(cfgDict['ppmm'])
# roi_size in config file is in mm. We convert it to pixel coordinates and round off.
roi_size = [int(round(x)) for x in np.array([float(i) for i in cfgDict['roi_size'].split(',')]) * ppmm]
total_trials = int(cfgDict['total_trials'])
maxTrialDur = int(cfgDict['max_trial_dur'])
rest_duration = int(cfgDict['initial_rest_dur'])
successrest_duration = int(cfgDict['success_rest_dur'])
failrest_duration = int(cfgDict['fail_rest_dur'])
bregma = list(map(int, cfgDict['bregma'].split(', ')))
br = clh.Position(bregma[0], bregma[1])
# get seep pixel locations(in mm) from config file and parse to a dictionary
cfgDict['seeds_mm'] = json.loads(cfgDict['seeds_mm'])
seeds = clh.generate_seeds(br, cfgDict['seeds_mm'] , ppmm, 'u')
normal_seeds_x_R = []
normal_seeds_y_R = []
normal_seeds_x_L = []
normal_seeds_y_L = []

runningAvg = np.zeros((res[0], res[1], 3))
runningImgQueue = deque(maxlen=10)
runningImgQueue.append(np.zeros((res[0], res[1], 3)))
runningImgQuROI = []

for name in roi_names:
    seed = seeds[name]
    rois.append(roi_manager.Rect(name,
                                 x= int(seed['ML']-roi_size[0]/2),
                                 y= int(seed['AP']-roi_size[1]/2),
                                 w= roi_size[0],h=roi_size[1],
                                 color=[0,0,255]))
    runningImgQuROI.append(deque(maxlen=dffHistory))
    runningImgQuROI[-1].append(np.zeros((roi_size[0], roi_size[1], 3)))

audio_tr_prob = float(config.get(cfg, 'audio'))
n_tones = int(config.get(cfg, 'n_tones'))
audiodelay = int(config.get(cfg, 'audio_delay'))
freqQue = deque(maxlen=(audiodelay*fr)+1)
freqQue.extend([1000]*(audiodelay*fr))
rewarddelay = int(config.get(cfg, 'reward_delay'))
relAvgDffQue = deque(maxlen=(rewarddelay*fr)+1)
relAvgDffQue.extend([0]*(rewarddelay*fr))

reward_threshold = float(config.get(cfg, 'reward_threshold'))
adaptive_threshold = int(config.get(cfg, 'adaptive_threshold'))
sessionType = 'NANR'

# Create I2C bus.
i2c = busio.I2C(board.SCL, board.SDA)

# Create MPR121 object.
touchCap = adafruit_mpr121.MPR121(i2c)

#list of sessions. session_name,reward_threshold
sessions = [('S1',reward_threshold)]

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

dff_bins = np.linspace(0, reward_threshold, len(freqs)-2)
## set lower and upper limits too high so dff indexing never goes out of range
dff_bins = np.insert(dff_bins, 0, np.NINF)
dff_bins = np.append(dff_bins, np.inf)
#dff_bins[0] = -200
#dff_bins[-1] = 200

ledBehavior = LED(17)
ledReward = LED(27)
ledFailTrial = LED(12)
ledLightTTL = LED(21)
ledLightTTL.off()
ts_trigger = 0
ts_detection = 0
run_threads = True
runPreview = True
runRecording = True


mouse_id = input("Please enter mouse ID: ")

WINDOW_NAME = 'Closed loop'
DFF_WINDOW = 'Corrected DFF'

cvui.init(WINDOW_NAME)
cvui.init(DFF_WINDOW)
roi_manager.init(roi_manager.annots, rois, WINDOW_NAME, res[0], res[1])
cv2.setMouseCallback(WINDOW_NAME, roi_manager.dragcircle, roi_manager.annots)
#blink 'led' once for 't' length
def timed_led(led, t):
    global fr, run_threads
    led.on()
    while run_threads:
        time.sleep(1/fr)
    led.off()

def blink_led(led, t):
    led.on()
    time.sleep(t)
    led.off()

#continuously blink 'led' for 't' length and 'gap' between blinks
def cont_blink_led(led, t, gap):
    print("blink led continuous")
    global ts_trigger
    global run_threads
    while(run_threads):
        ts_trigger = time.time()
        blink_led(led,t)
        time.sleep(gap)
        
# Blink the red LED every 3 seconds - no longer needed
#tmr = Timer(3,cont_blink_led, args=(ledBehaviorStart, 0.1, 2))
#tmr.start()

summary_filename = data_root + os.sep + config.get(cfg, 'summary_file')
summary_exists = os.path.isfile(summary_filename)
summaryfile = open (summary_filename, 'a', encoding="utf-8")
headers = [col.strip() for col in config.get(cfg, 'summary_header').split(',')]
writer = csv.DictWriter(summaryfile, delimiter=',', lineterminator='\n',fieldnames=headers)
if not summary_exists:
    writer.writeheader()  # file doesn't exist yet, write a header

print("Start preview\n\n")
# initialize the camera and grab a reference to the raw camera capture
exec('from ' + vidSourceName + ' import ' + vidSourceName)
# initialize the camera and grab a reference to the raw camera capture
vs = CameraFactory(eval(vidSourceName+'(cfgDict).start()'))

# allow the camera to warmup
time.sleep(2)

ledLightTTL.on()
while runPreview:
    start = time.time()
    #image = frame.array
    img = vs.get_image()
    
    if touchCap[0].value:
         print('Pin 0 touched!')
#     print('analog gain: ' + str(eval(str(vs.camera_factory.camera.analog_gain))))
#     print('digital gain: ' + str(eval(str(vs.camera_factory.camera.digital_gain))))
#     print('ISO: ' + str(eval(str(vs.camera_factory.camera.iso))))
    # select only green channel
    #img = img[:,:,1]
    image = img.copy()
    #image = image[:,:,1:3]
    
    runningImgQueue.append(image)
    runningAvg = np.mean(runningImgQueue, axis=0)
    
    dff = image - runningAvg
    dff = np.divide(dff, runningAvg, out=np.zeros_like(dff), where=runningAvg!=0)
    dffGreen = dff[:,:,1]
    dffBlue = dff[:,:,0]
    dffCorrected = dffGreen - dffBlue
    #dffCorrected = dff
    #dffCorrected = dffGreen
    
    #dffCorrected[dffCorrected >0.05] =1
    #print("Max blue  is: ",np.max(dffBlue))
    lumBlue = image[res[0]//2][0:,0]
    lumGreen = image[res[0]//2][0:,1]
    lumBlue[0] = 1
    lumGreen[0] = 1
    cvui.sparkline(image, lumBlue, 0, res[0]-100, res[0], 50, 0x00ff00)
    cvui.sparkline(image, lumGreen, 0, res[0]-100, res[0], 50, 0x00bb00)
    cvui.printf(image, 5, 5, "max lumBlue: %d", max(lumBlue))
    cvui.printf(image, 5, 20, "max lumGreen: %d", max(lumGreen))
    
    # Render the roi
    for roi in rois:
        cv2.rectangle(image, (roi.x, roi.y), (roi.x+roi.w, roi.y+roi.h), roi.color, 1)
        cropRoi = dffCorrected[roi.y : roi.y + roi.h, roi.x : roi.x + roi.w]
        roi.avgDff = np.sum(cropRoi)/roi.area()
#    for roi, imQ in zip(rois,runningImgQuROI):
#        imROI = image[roi.y : roi.y + roi.h, roi.x : roi.x + roi.w,:]
#        
#        runAvg = np.mean(imQ, axis=0)
#        dffROI = imROI - runAvg
#        dffROI = np.divide(dffROI, runAvg, out=np.zeros_like(dffROI), where=runAvg!=0)
#        dffGreen = dffROI[:,:,1]
#        dffBlue = dffROI[:,:,0]
#        dffCorrected = dffGreen - dffBlue
#        imQ.append(imROI)
#        cv2.rectangle(image, (roi.x, roi.y), (roi.x+roi.w, roi.y+roi.h), roi.color, 1)
#        #cv2.rectangle(fgmaskBGR, (roi.x, roi.y), (roi.x+roi.w, roi.y+roi.h), roi.color, 2)
#        #cropRoi = dffCorrected[roi.y : roi.y + roi.h, roi.x : roi.x + roi.w]
#        roi.avgDff = np.sum(dffCorrected)/roi.area()
    
    #mark seed locations in raw image for display
    image[br.row, br.col] = 255
    for seedname in seeds:
        image[seeds[seedname]['AP'], seeds[seedname]['ML']] = 255
        if "_R" in seedname:
            normal_seeds_x_R.append(seeds[seedname]['ML'])
            normal_seeds_y_R.append(seeds[seedname]['AP'])
        else:
            normal_seeds_x_L.append(seeds[seedname]['ML'])
            normal_seeds_y_L.append(seeds[seedname]['AP'])
    #mark seed locations in dff map for display
    dffCorrected[br.row, br.col] = 255
    for seedname in seeds:
        dffCorrected[seeds[seedname]['AP'], seeds[seedname]['ML']] = 255
    relAvgDff = rois[0].avgDff
    #print("RelAvgDff is: ",relAvgDff)
    #combined = np.concatenate((image, dffCorrected), axis=1)
    # This function must be called *AFTER* all UI components. It does
    # all the behind the scenes magic to handle mouse clicks, etc.
    cvui.update()
    cvui.imshow(WINDOW_NAME, cv2.resize(image,(512,512)));
    cvui.imshow(DFF_WINDOW, cv2.resize(dffCorrected, (512,512)));
    
    # Press Esc or Ctrl-C to stop the program
    k = cv2.waitKey(1)
    if k == 27:
        runPreview = False
        break
    time.sleep(max(1./fr - (time.time() - start), 0))
    #vs.camera_factory.updatesleep(max(0.5/(fr) - (time.time() - start), 0.0))

for roi in rois:
    config.set(cfg, roi.name,str(roi.x)+','+str(roi.y)+','+str(roi.w)+','+str(roi.h))
#co save the configuration used in current directory
with open(configPath, 'w', encoding="utf-8") as f:
        config.write(f)

cv2.destroyWindow(DFF_WINDOW)
ledLightTTL.off()
#vs.stop()
time.sleep(2)
# Get the current time and initialize the project folder
tm = datetime.now()
data_root = data_root + os.sep + mouse_id + os.sep
data_root = data_root + str(tm.year) + \
            format(tm.month, '02d') + \
            format(tm.day, '02d') + \
            format(tm.hour, '02d') + \
            format(tm.minute, '02d') + \
            format(tm.second, '02d')
if not os.path.exists(data_root):
        print("Creating data directory: ",data_root)
        os.makedirs(data_root)

for session, reward_threshold in sessions:
    #breakpoint()
    
    if runRecording:
        #runningImgQueue.clear()
        #runningImgQueue.append(np.zeros((res[0], res[1], 3)))
        totRewards = 0;
        #send TTL to behavior PI to start recording
        ledBehavior.on()
        # Get the current time and initialize the project folder
        tm = datetime.now()
        session_root = data_root + os.sep + session
        if not os.path.exists(session_root):
                print("Creating data directory: ",session_root)
                os.makedirs(session_root)
        # initialize the camera and grab a reference to the raw camera capture
        #vs = PiVideoStream(resolution=res, framerate=fr, save_path = data_root + os.sep + "raw_frames.jpg").start()
        #vs = CameraFactory(eval(vidSourceName+'(cfgDict).start()'))
        # allow the camera to warmup
        time.sleep(2)
        image = vs.get_image()
        rows, cols, channels = image.shape
        time.sleep(1) # Let behavior recording start first
        
        config.set(cfg, 'reward_threshold',str(reward_threshold))
        config.set('configsection', 'config', cfg)
        # save the configuration used in target data directory
        with open(session_root + '/config.ini', 'w', encoding="utf-8") as f:
                config.write(f)
        image_hdf5_path = session_root + os.sep + image_stream_filename
        image_hdf5_file = tables.open_file(image_hdf5_path, mode='w')
        image_storage = image_hdf5_file.create_earray(image_hdf5_file.root,
                                                      'raw_images',
                                                      tables.Atom.from_dtype(image.dtype),
                                      shape=(0, res[0], res[1], channels))
#        frame_storage = image_hdf5_file.create_earray(image_hdf5_file.root,
#                                                      'frame_n',
#                                                      tables.Atom.from_dtype(totRewards), (0,))
        logFileName = session_root + os.sep + "VideoTimestamp.txt"
        logFile = open(logFileName, 'w', encoding="utf-8")
        logFile.write('frame' + '\t' +
                      'time' + '\t' +
                      roi_names[0]+'dff' + '\t' +
                      'freq' + '\t' +
                      'rew_threshold' + '\t' +
                      'reward' + '\t' +
                      'trial' + '\t' +
                      'audio' + '\t' +
                      'lick' + '\n')
        print("Start recording\n")
        
    ledLightTTL.on()
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
        #print('AWB gain: ' + str(vs.camera_factory.camera.awb_gains))
        reward = 0
        touch = 0
        
        if touchCap[0].value:
             print('Lick!')
             touch = 1
        
        if image is None:
            break
        # select only green channel
        #image = image[:,:,1]
        fps.update()
        
        for roi, imQ in zip(rois,runningImgQuROI):
            imROI = image[roi.y : roi.y + roi.h, roi.x : roi.x + roi.w,:]
            
            runAvg = np.mean(imQ, axis=0)
            dffROI = imROI - runAvg
            dffROI = np.divide(dffROI, runAvg, out=np.zeros_like(dffROI), where=runAvg!=0)
            dffGreen = dffROI[:,:,1]
            dffBlue = dffROI[:,:,0]
            dffCorrected = dffGreen - dffBlue
            imQ.append(imROI)
            roi.avgDff = np.sum(dffCorrected)/roi.area()
        
        relAvgDff = rois[0].avgDff
        relAvgDffQue.append(relAvgDff)
        relAvgDffDelayed = relAvgDffQue.popleft()
        
        fq = int(freqs[np.searchsorted(dff_bins, relAvgDff)])
        freqQue.append(fq)
        freq = freqQue.popleft()
        #freq = int(freqs[nTrial-1])
        if runTrial:
            tr = nTrial
            rest_duration = failrest_duration
            # relative  dff should be greater than the threshold
            # and current reward should be atleast 1 sec apart from previous
            if relAvgDffDelayed > reward_threshold:
                reward = 1
                totRewards += 1
                rewardsInEpoch += 1
                #End the trial since reward obtained
                #rest = True
                #runTrial = False
                print("RelAvgDff is: ",relAvgDffDelayed)
                print("Rew threshold is: ",reward_threshold)
                print('total rewards: ' + str(totRewards), end="\n")
                t2 = Thread(target=blink_led, args=(ledReward, 0.3))
                t2.start()
                lastRewTime = time.time()
                runTrial = False
                rest_duration = successrest_duration
            
            elif time.time() - trialTimer >= maxTrialDur:
                runTrial = False
                reward = -1
                t2 = Thread(target=blink_led, args=(ledFailTrial, 0.5))
                t2.start()
            restTimer = time.time()
            #play_tone(freq, 0.1, stream)
            sinsource.frequency = freq
            #print("Freq is: ",freq)
        else:
            if not rest: #trial just ended
                sinsource.stop()
                print('Trial end')
            rest = True
            runTrial = False
            
            if rest and time.time() - restTimer < rest_duration:
                freq = 0
                tr = 0
                audio = 0
                trialTimer = time.time()
                
            else:
                rest = False
                runTrial = True
                audio = audio_tr_arr[tr]
                if audio:
                    austream = get_output(channels=n_aud_ch, rate=au_rate, buffersize=128)
                    sinsource = SineSource(austream, freq)
                    sinsource.start()
    
                    # check if adaptive threshold is enabled every 30 seconds
        if adaptive_threshold and time.time() - rewardThreshTimer > 30:
            # if more than 1 reward dispensed in last epoch, increase the threshold
            if rewardsInEpoch > 1:
                reward_threshold += 0.002
            # if no reward was dispensed in last epoch, decrease the threshold
            if rewardsInEpoch ==0:
                reward_threshold -= 0.002

            rewardsInEpoch = 0
            rewardThreshTimer = time.time()
            # regenerate the dff bins to frequency mappings
            dff_bins = np.linspace(-reward_threshold, reward_threshold, len(freqs)-2)
            dff_bins = np.insert(dff_bins, 0, np.NINF)
            dff_bins = np.append(dff_bins, np.inf)
        
        if tr ==0 and runTrial:
            nTrial += 1
            print("Trial: ",nTrial)
        
        image_storage.append(image[None])
#        frame_storage.append(fps._numFrames)
        sttime = datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')
        logFile.write(str(fps._numFrames) + '\t' +
                      sttime + '\t' +
                      str(rois[0].avgDff) + '\t' +
                      str(freq) + '\t' +
                      str(reward_threshold) + '\t' +
                      str(reward) + '\t' +
                      str(tr) + '\t' +
                      str(audio) + '\t' +
                      str(touch) + '\n')
        
        #if cvui.button(image, 100, 230, "&Quit"):
        #    break
        
        #cvui.imshow(WINDOW_NAME, combined);
        cvui.imshow(WINDOW_NAME, cv2.resize(image,(512,512)));
        #cvui.imshow(DFF_WINDOW, cv2.resize(runningImgQuROI[0],(512,512)))
        # Press Esc or Ctrl-C to stop the program
        k = cv2.waitKey(1)
        if k == 27:
            break
        
        time.sleep(max(1./fr - (time.time() - start), 0))
        #vs.camera_factory.updatesleep(max(0.5/(fr) - (time.time() - start), 0.0))
        
    ledLightTTL.off()
    fps.stop()
    vs.stop()
    print('total rewards: ' + str(totRewards))
    
    time.sleep(1) # let vs stop and behavior recording capture light off
    ledBehavior.off()
    
    image_hdf5_file.close()
    logFile.close()

    writer.writerow({headers[0]: mouse_id,
                     headers[1]: session,
                     headers[2]: session_root,
                     headers[3]: fps._start,
                     headers[4]: fps._end,
                     headers[5]: fps.elapsed(),
                     headers[6]: fps.fps(),
                     headers[7]: dffHistory,
                     headers[8]: n_tones,
                     headers[9]: reward_threshold,
                     headers[10]: totRewards,
                     headers[11]: audio_tr_prob,
                     headers[12]: sessionType})
    print("[INFO] elasped time: {:.2f}".format(fps.elapsed()))
    print("[INFO] approx. FPS: {:.2f}\n\n".format(fps.fps()))

run_threads = False

summaryfile.close()
#audPin.set_enable (0,1)
#stream.close()
sinsource.stop()
#p.terminate()
cv2.destroyAllWindows()


