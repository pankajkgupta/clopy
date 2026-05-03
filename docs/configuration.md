# Configuration Reference

Complete reference for all configuration options in `config.ini`.

## File Structure

```ini
[configsection]
config = section_name

[section_name]
# Camera settings
# Processing settings
# Trial settings
# Reward settings
```

## CLNF Configuration Sections

### raspicambrain_reward_punish_1roi

Single ROI configuration for CLNF experiments.

```ini
[raspicambrain_reward_punish_1roi]
vid_source = PiCameraStream
data_root = /media/pi/data/cl_brain/
raw_image_file = image_stream.hdf5
resolution = 256, 256
framerate = 15
awb_mode = off
shutter_speed = 60000
iso = 800
sensor_mode = 7
dff_history = 6
ppmm = 25.6
bregma = 135, 128
seeds_mm = {"ALM": {"ML": 1.5, "AP": 2.5}, ...}
roi = M1_L
roi_size = 0.3, 0.3
audio = 1
n_tones = 18
audio_delay = 0
reward_delay = 1
reward_threshold = 0.075
adaptive_threshold = 1
total_trials = 60
max_trial_dur = 30
success_rest_dur = 10
fail_rest_dur = 15
initial_rest_dur = 30
summary_file = expt_1roi_trial_summary.csv
summary_header = mouse_id, session, data_path, start, end, duration, fps, dff_history, n_tones, reward_threshold, rewards, audio, session_type
```

### raspicambrain_reward_punish_2roi

Dual ROI configuration for CLNF experiments.

```ini
[raspicambrain_reward_punish_2roi]
vid_source = PiCameraStream
data_root = /media/pi/data/cl_brain/
raw_image_file = image_stream.hdf5
resolution = 256, 256
framerate = 15
awb_mode = off
shutter_speed = 60000
iso = 800
sensor_mode = 7
dff_history = 6
ppmm = 25.6
bregma = 135, 128
seeds_mm = {"ALM": {"ML": 1.5, "AP": 2.5}, ...}
roi_operation = BC_L-BC_R
roi_size = 0.3, 0.3
audio = 1
n_tones = 18
audio_delay = 0
reward_delay = 1
reward_threshold = 0.1
adaptive_threshold = 0
total_trials = 60
max_trial_dur = 30
success_rest_dur = 5
fail_rest_dur = 10
initial_rest_dur = 30
summary_file = expt_2roi_trial_summary.csv
summary_header = mouse_id, session, data_path, start, end, duration, fps, dff_history, n_tones, reward_threshold, rewards, audio, session_type
```

## CLMF Configuration Sections

### sentech_dlclive

Configuration for CLMF experiments with DeepLabCut-Live.

```ini
[sentech_dlclive]
vid_source = SentechCameraStream
data_root = /media/murphy/fast_storage/data/
video_file = behavior_video.mp4
resolution = 640, 320
framerate = 30
awb_mode = off
shutter_speed = 60000
iso = 800
joint_history_sec = 5
ppmm = 3.5
bregma = 108, 128
control_point = fl_l
start_center = 590,110
start_radius_mm = 4
target_center = 590,80
target_radius_mm = 6
speed_threshold = 20
audio = 1
n_tones = 18
audio_delay = 0
reward_delay = 1
reward_threshold_mm = 5
adaptive_threshold = 0
total_trials = 10
max_trial_dur = 30
success_rest_dur = 10
fail_rest_dur = 10
initial_rest_dur = 30
dlc_model_path = /path/to/dlc-model/
summary_file = expt_dlclive_summary.csv
summary_header = mouse_id, session, data_path, start, end, duration, fps, n_tones, start_radius, target_radius, rewards, audio, session_type
```

## Parameter Descriptions

### Camera Settings

| Parameter | Description | Example |
|-----------|-------------|---------|
| `vid_source` | Camera driver class | `PiCameraStream`, `SentechCameraStream` |
| `resolution` | Image width, height | `256, 256` or `640, 320` |
| `framerate` | Frames per second | `15`, `30`, `60` |
| `awb_mode` | Auto white balance | `off`, `auto` |
| `shutter_speed` | Camera shutter speed (μs) | `60000`, `600000` |
| `iso` | Camera ISO | `800` |
| `sensor_mode` | PiCamera sensor mode | `7` |

### Processing Settings

| Parameter | Description | Example |
|-----------|-------------|---------|
| `dff_history` | Frames for baseline calculation | `6` |
| `ppmm` | Pixels per millimeter | `25.6`, `3.5` |
| `bregma` | Bregma landmark (row, col) | `135, 128` |
| `seeds_mm` | Brain region coordinates (JSON) | `{"ALM": {"ML": 1.5, "AP": 2.5}}` |

### ROI Settings

| Parameter | Description | Example |
|-----------|-------------|---------|
| `roi` | Single ROI name | `M1_L` |
| `roi_operation` | Dual ROI operation | `BC_L-BC_R`, `M1_L+ALM` |
| `roi_size` | ROI size in mm | `0.3, 0.3` |
| `start_center` | Start ROI center (pixels) | `590, 110` |
| `start_radius_mm` | Start ROI radius (mm) | `4` |
| `target_center` | Target ROI center (pixels) | `590, 80` |
| `target_radius_mm` | Target ROI radius (mm) | `6` |

### Audio Settings

| Parameter | Description | Example |
|-----------|-------------|---------|
| `audio` | Enable audio feedback | `0` or `1` |
| `n_tones` | Number of frequency bins | `18` |
| `audio_delay` | Audio delay (frames) | `0` |

### Reward Settings

| Parameter | Description | Example |
|-----------|-------------|---------|
| `reward_threshold` | ΔF/F threshold (CLNF) | `0.075` |
| `reward_threshold_mm` | Distance threshold (CLMF) | `5` |
| `speed_threshold` | Speed threshold (CLMF) | `20` |
| `reward_delay` | Reward delay (frames) | `1` |
| `adaptive_threshold` | Enable adaptive threshold | `0` or `1` |

### Trial Settings

| Parameter | Description | Example |
|-----------|-------------|---------|
| `total_trials` | Number of trials | `60` |
| `max_trial_dur` | Max trial duration (sec) | `30` |
| `success_rest_dur` | Rest after success (sec) | `10` |
| `fail_rest_dur` | Rest after failure (sec) | `15` |
| `initial_rest_dur` | Initial rest period (sec) | `30` |

### DeepLabCut Settings (CLMF)

| Parameter | Description | Example |
|-----------|-------------|---------|
| `dlc_model_path` | Path to DLC model | `/path/to/model/` |
| `control_point` | Body part to track | `fl_l` |
| `joint_history_sec` | Stability history (sec) | `5` |

### Data Output Settings

| Parameter | Description | Example |
|-----------|-------------|---------|
| `data_root` | Base data directory | `/media/pi/data/` |
| `raw_image_file` | HDF5 file name | `image_stream.hdf5` |
| `video_file` | Video file name | `behavior_video.mp4` |
| `summary_file` | Summary CSV name | `expt_summary.csv` |
| `summary_header` | CSV column names | (comma-separated) |

## Brain Region Coordinates

Default seed positions (in mm from bregma):

| Region | ML | AP |
|--------|----|----|
| ALM | 1.5 | 2.5 |
| M1 | 1.86 | 0.64 |
| M2 | 0.87 | 1.42 |
| FL | 2.45 | -0.57 |
| HL | 1.69 | -1.15 |
| BC | 3.46 | -1.73 |
| V1 | 2.52 | -3.77 |
| RS | 0.62 | -2.89 |

## Editing Configuration

1. Open `config.ini` in a text editor
2. Find the appropriate section
3. Modify values as needed
4. Save the file

!!! tip "ROI Position Adjustment"
    After running the preview, ROI positions are saved back to `config.ini`. You can also manually specify coordinates:

```ini
m1_l = 76,114,8,8
```
Format: `x, y, width, height`