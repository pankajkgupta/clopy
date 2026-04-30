# CLMF - Closed-Loop Movement Feedback

CLMF uses real-time pose estimation (DeepLabCut-Live) to provide audio feedback correlated with movement speed of a specified body part.

## How It Works

```
Camera → Pose Estimation → Speed Calculation → Threshold Check → Audio Feedback + Reward
```

1. **Pose Tracking**: DeepLabCut-Live infers body part positions
2. **Speed Calculation**: Compute movement speed between frames
3. **Feedback**: Map speed to audio tone frequency
4. **Reward**: Deliver water when speed exceeds threshold

## Running CLMF Experiments

```bash
python behavior/cla_dlc_trials_speed.py
```

## Configuration Parameters

### Required Settings (config.ini)

```ini
[sentech_dlclive]
vid_source = SentechCameraStream
data_root = /path/to/data/
dlc_model_path = /path/to/dlc-model/
resolution = 640, 320
framerate = 30
joint_history_sec = 5
ppmm = 3.5
bregma = 108, 128
control_point = fl_l
start_center = 590, 110
start_radius_mm = 4
target_center = 590, 80
target_radius_mm = 6
speed_threshold = 20
audio = 1
n_tones = 18
reward_threshold_mm = 5
adaptive_threshold = 0
total_trials = 10
max_trial_dur = 30
success_rest_dur = 10
fail_rest_dur = 10
initial_rest_dur = 30
```

### Parameter Reference

| Parameter | Description | Typical Value |
|-----------|-------------|---------------|
| `vid_source` | Camera driver | `SentechCameraStream` |
| `dlc_model_path` | DeepLabCut model directory | `/path/to/model/` |
| `resolution` | Image resolution | `640, 320` |
| `framerate` | Acquisition frame rate | `30` |
| `joint_history_sec` | History for stability check | `5` |
| `ppmm` | Pixels per millimeter | `3.5` |
| `control_point` | Body part to track | `fl_l`, `fl_r`, etc. |
| `start_center` | Start ROI center (pixels) | `590, 110` |
| `start_radius_mm` | Start ROI radius (mm) | `4` |
| `target_center` | Target ROI center (pixels) | `590, 80` |
| `target_radius_mm` | Target ROI radius (mm) | `6` |
| `speed_threshold` | Speed threshold (pixels/frame) | `20` |
| `n_tones` | Number of audio frequencies | `18` |
| `reward_threshold_mm` | Distance threshold for reward | `5` |
| `adaptive_threshold` | Auto-adjust threshold | `0` or `1` |

## Body Parts (DeepLabCut)

The system tracks all body parts defined in your DeepLabCut model. Common options:

- `fl_l` - Left forelimb
- `fl_r` - Right forelimb
- `hl_l` - Left hindlimb
- `hl_r` - Right hindlimb

## Trial Initialization Options

### Option 1: Stability-based (Default)

Trials start when the tracked body part is stable (not moving) for `joint_history_sec`:

```python
if np.mean(cj_speed_hist_que) < 1:  # stable
    start_trial()
```

### Option 2: Start ROI-based

Trials start when the body part is within the start ROI:

```python
if cj_start_dist < start_radius:
    start_trial()
```

### Option 3: Immediate (After Rest)

Trials start immediately after rest period ends:

```python
# Uncomment in code:
rest = False
runTrial = True
```

## Audio Feedback Mapping

Movement speed is mapped to audio frequency:

```
Speed = 0    →  1000 Hz (low tone)
Speed = threshold → 1000 × 2^(n_tones/4) Hz (high tone)
```

## Session Types

Configure in code:

```python
# Normal audio + normal reward
sessionType = clh.SessionType.normal_audio_normal_reward

# Single audio + normal reward (constant tone)
sessionType = clh.SessionType.single_audio_normal_reward

# No audio + random reward (control)
sessionType = clh.SessionType.no_audio_random_reward

# Normal audio + no reward (just feedback)
sessionType = clh.SessionType.normal_audio_no_reward

# No audio + no reward (control)
sessionType = clh.SessionType.no_audio_no_reward
```

## Adaptive Thresholding

When `adaptive_threshold = 1`:

- Every 30 seconds, check rewards in epoch
- If > 1 reward: increase speed threshold by 2
- If 0 rewards: decrease speed threshold by 2 (minimum 2)

## Hardware Setup

### Required Components

- Nvidia Jetson Orin (or equivalent GPU)
- Sentech camera (or USB3 camera)
- Arduino (for TTL and LED control)
- Water reward valve
- Audio speaker

### Arduino Pinout

| Pin | Function |
|-----|----------|
| 13 | Brain TTL (output) |
| 7 | Reward LED |
| 12 | Fail LED |
| 40 | Light TTL |

### GPIO (RPi GPIO numbering)

| Pin | Function |
|-----|----------|
| 13 | Brain TTL |
| 7 | Reward LED |
| 12 | Fail LED |
| 40 | Light TTL |

## Data Output

### Log File Format

```
frame	time	cj_speed	freq	reward	trial	audio	lick	pose_x	pose_y	likelihood...
```

### Pose Data

All DeepLabCut body parts are logged with x, y, and likelihood values.

### Summary CSV

```
mouse_id, session, data_path, start, end, duration, fps, n_tones, start_radius, target_radius, rewards, audio, session_type
```

## Example Workflow

1. Train DeepLabCut model on your setup
2. Configure `dlc_model_path` in `config.ini`
3. Set `control_point` to the body part to track
4. Calibrate `ppmm` (pixels per millimeter)
5. Run: `python behavior/cla_dlc_trials_speed.py`
6. Enter mouse ID when prompted
7. Adjust ROI positions in preview window
8. Press `Esc` to start session
9. Monitor real-time pose and speed
10. Press `Esc` to end or wait for completion

## Video Recording

The system records behavior video during sessions:

- Format: MP4
- Resolution: Matched to camera settings
- Frame rate: Matched to acquisition

## Troubleshooting

See [Troubleshooting Guide](troubleshooting.md) for common issues, including:
- DeepLabCut-Live initialization
- GPU memory management
- Pose estimation quality