# Troubleshooting Guide

Solutions to common issues encountered when running CLoPy experiments.

## Installation Issues

### PyAudio Installation Fails

**Symptom**: `pip install PyAudio` fails

**Solution**:
=== "Linux"
    ```bash
    sudo apt-get install portaudio19-dev python3-pyaudio
    pip3 install --no-cache-dir PyAudio
    ```

=== "Windows"
    Download precompiled wheel from [Christophe Gauthier's site](https://people.csail.mit.edu/hubert/pyaudio/)

### Audiostream Installation Fails

**Symptom**: `python setup.py install` fails

**Solution**:
```bash
# Install dependencies first
sudo apt-get install libsdl1.2-dev libsdl-mixer1.2-dev
pip3 install cython==0.29.21 kivy
```

### DeepLabCut-Live Import Error

**Symptom**: `ImportError: No module named 'dlclive'`

**Solution**:
```bash
# Check TensorFlow version compatibility
pip install tensorflow==2.10.0  # or compatible version
pip install deeplabcut-live
```

## Camera Issues

### PiCamera Not Detected

**Symptom**: Camera not found or black images

**Solution**:
1. Enable camera interface:
   ```bash
   sudo raspi-config
   ```
   → Interface Options → Camera → Enable

2. Check camera connection:
   ```bash
   raspistill -o test.jpg
   ```

3. Verify device:
   ```bash
   ls -la /dev/video*
   ```

### Sentech Camera Not Working

**Symptom**: No image or errors

**Solution**:
1. Check USB connection:
   ```bash
   v4l2-ctl --list-devices
   ```

2. Test with GST:
   ```bash
   gst-launch-1.0 v4l2src device=/dev/video0 ! videoconvert ! ximagesink
   ```

3. Check resolution compatibility in `config.ini`

## CLNF-Specific Issues

### ΔF/F Values Too High/Low

**Symptom**: Activity values outside expected range

**Solution**:
- Adjust `dff_history` (more frames = more stable baseline)
- Check `roi_size` is appropriate
- Verify `ppmm` calibration
- Adjust `reward_threshold` accordingly

### ROI Not Visible in Preview

**Symptom**: Cannot see ROI rectangles

**Solution**:
1. Check `seeds_mm` contains valid brain region
2. Verify `bregma` coordinates are correct
3. Ensure `roi_size` is not zero
4. Check camera is imaging correct area

### Audio Not Playing

**Symptom**: No audio feedback during trials

**Solution**:
1. Check `audio = 1` in config
2. Verify speaker is connected
3. Test audio independently:
   ```python
   from audiostream import get_output
   from audiostream.sources.wave import SineSource
   stream = get_output(channels=1, rate=44100)
   source = SineSource(stream, 1000)
   source.start()
   ```

### Lick Detection Not Working

**Symptom**: No lick detection even when mouse licks

**Solution**:
1. Check I2C connection:
   ```bash
   sudo i2cdetect -y 1
   ```
   Should show device at `0x5a`

2. Verify wiring (SCL, SDA, VCC, GND)

3. Test MPR121:
   ```python
   import board
   import busio
   import adafruit_mpr121
   i2c = busio.I2C(board.SCL, board.SDA)
   cap = adafruit_mpr121.MPR121(i2c)
   print(cap[0].value)  # Should change on touch
   ```

## CLMF-Specific Issues

### DeepLabCut-Live Initialization Fails

**Symptom**: Error loading model

**Solution**:
1. Verify model path exists:
   ```bash
   ls -la /path/to/dlc-model/
   ```

2. Check model format (should contain `*.pb` files)

3. Verify TensorFlow version:
   ```bash
   pip list | grep tensorflow
   ```

4. Try with display=False:
   ```python
   dlc_live = DLCLive(model_path, display=False)
   ```

### Pose Estimation Very Slow

**Symptom**: Low FPS, cannot keep up with camera

**Solution**:
1. Use GPU acceleration (Jetson Orin)
2. Reduce input resolution
3. Use smaller model
4. Check for memory issues:
   ```python
   import gc
   gc.collect()
   ```

### Body Part Not Tracking

**Symptom**: No pose data or NaN values

**Solution**:
1. Verify `control_point` exists in model:
   ```python
   print(dlc_live.cfg['all_joints_names'])
   ```

2. Check model was trained on that body part

3. Ensure body part is visible in frame

4. Adjust confidence threshold if needed

### Speed Values Too High/Low

**Symptom**: Speed always above/below threshold

**Solution**:
- Calibrate `ppmm` (pixels per millimeter)
- Adjust `speed_threshold` in config
- Check camera resolution matches config

## Data Issues

### HDF5 File Not Writing

**Symptom**: Error creating or writing to `.hdf5` file

**Solution**:
1. Check write permissions:
   ```bash
   ls -la /path/to/data/
   ```

2. Ensure directory exists:
   ```python
   os.makedirs(data_root, exist_ok=True)
   ```

3. Check disk space

### Log File Format Error

**Symptom**: Cannot read VideoTimestamp.txt

**Solution**:
- Check delimiter (tab-separated)
- Verify column names match expected
- Handle missing values appropriately

## Performance Issues

### Low Frame Rate

**Symptom**: FPS much lower than configured

**Solution**:
1. Reduce processing load:
   - Smaller ROI
   - Lower resolution
   - Fewer audio tones

2. Optimize code:
   - Use numpy operations
   - Avoid unnecessary copies

3. Check hardware:
   - SD card speed (use fast card)
   - CPU throttling (check `vcgencmd get_throttled`)

### Memory Errors

**Symptom**: `MemoryError` or system freezes

**Solution**:
1. Reduce buffer sizes:
   - Smaller `dff_history`
   - Smaller `joint_history_sec`

2. Clear caches:
   ```python
   import gc
   gc.collect()
   ```

3. Use smaller data types where possible

## Session Issues

### Session Won't Start

**Symptom**: Preview closes immediately or won't start

**Solution**:
1. Check all required config parameters
2. Verify camera is providing frames
3. Ensure ROIs are within image bounds
4. Check for errors in terminal output

### Early Session Termination

**Symptom**: Session ends before all trials complete

**Solution**:
1. Check for `Esc` key press
2. Verify no Python errors
3. Check disk space for data recording

## Hardware Issues

### LEDs Not Lighting

**Symptom**: No LED response

**Solution**:
1. Check GPIO connections
2. Verify resistor values (330Ω recommended)
3. Test LED independently
4. Check GPIO numbering (Board vs BCM)

### Water Reward Not Dispensing

**Symptom**: No water delivery

**Solution**:
1. Check valve wiring
2. Verify Arduino communication:
   ```python
   board.digitalWrite(ledReward, "HIGH")
   ```
3. Check water level
4. Inspect valve for clogs

## Getting Help

If you continue to experience issues:

1. Check the [GitHub Issues](https://github.com/pankajkgupta/clopy/issues)
2. Review the paper methods section
3. Check your configuration against the examples
4. Verify all dependencies are compatible