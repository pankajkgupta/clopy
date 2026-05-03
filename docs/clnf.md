# CLNF - Closed-Loop Neurofeedback

CLNF uses real-time calcium imaging (ΔF/F) to provide audio feedback correlated with neural activity in a target brain region.

## How It Works

```
Camera → ΔF/F Calculation → Threshold Check → Audio Feedback + Reward
```

1. **Imaging**: Wide-field calcium imaging of cortical activity
2. **Processing**: Calculate ΔF/F (delta F over F) in real-time
3. **Feedback**: Map activity level to audio tone frequency
4. **Reward**: Deliver water when activity exceeds threshold

## Running CLNF Experiments

### Single ROI Experiment

```bash
python brain/cla_reward_punish_1roi.py
```

Monitors a single brain region. Reward triggered when activity exceeds threshold.

### Dual ROI Experiment

```bash
python brain/cla_reward_punish_2roi.py
```

Monitors two brain regions (can be extended to more number of regions) with mathematical operations such as:

- **Addition**: `ROI1 + ROI2`
- **Subtraction**: `ROI1 - ROI2`

Other operations such as multiplication, division etc. are supported and can be specified in the `config.ini` file.

## Configuration Parameters

### Required Settings (config.ini)

```ini
[raspicambrain_reward_punish_1roi]
vid_source = PiCameraStream
data_root = /path/to/data/
resolution = 256, 256
framerate = 15
dff_history = 6
ppmm = 25.6
bregma = 135, 128
roi = M1_L
roi_size = 0.3, 0.3
audio = 1
n_tones = 18
reward_threshold = 0.075
adaptive_threshold = 1
total_trials = 60
max_trial_dur = 30
success_rest_dur = 10
fail_rest_dur = 15
initial_rest_dur = 30
```

### Parameter Reference

| Parameter | Description | Typical Value |
|-----------|-------------|---------------|
| `vid_source` | Camera driver | `PiCameraStream` |
| `resolution` | Image resolution | `256, 256` |
| `framerate` | Acquisition frame rate | `15` |
| `dff_history` | Frames for baseline (seconds × framerate) | `6` (0.4s) |
| `ppmm` | Pixels per millimeter | `25.6` |
| `bregma` | Bregma landmark position | `135, 128` |
| `roi` | Target brain region | `M1_L`, `ALM`, etc. |
| `roi_size` | ROI dimensions (mm) | `0.3, 0.3` |
| `n_tones` | Number of audio frequencies | `18` |
| `reward_threshold` | ΔF/F threshold for reward | `0.075` |
| `adaptive_threshold` | Auto-adjust threshold | `0` or `1` |
| `total_trials` | Number of trials per session | `60` |
| `max_trial_dur` | Maximum trial duration (sec) | `30` |
| `success_rest_dur` | Rest after success (sec) | `10` |
| `fail_rest_dur` | Rest after failure (sec) | `15` |
| `initial_rest_dur` | Initial rest before first trial | `30` |

## Available Brain Regions

Pre-configured seed locations (in mm from bregma):

```python
seeds_mm = {
    "ALM": {"ML": 1.5, "AP": 2.5},
    "M1": {"ML": 1.8603, "AP": 0.64181},
    "M2": {"ML": 0.87002, "AP": 1.4205},
    "FL": {"ML": 2.4526, "AP": -0.5668},
    "HL": {"ML": 1.6942, "AP": -1.1457},
    "BC": {"ML": 3.4569, "AP": -1.727},
    "V1": {"ML": 2.5168, "AP": -3.7678},
    "RS": {"ML": 0.62043, "AP": -2.8858},
    # ... and more
}
```

## Audio Feedback Mapping

Neural activity is mapped to audio frequency:

```
ΔF/F = 0    →  1000 Hz (low tone)
ΔF/F = threshold → 1000 × 2^(n_tones/4) Hz (high tone)
```

The system uses quarter-octave frequency increments from 1-24 kHz.

## Session Types

Configure reward behavior in code:

```python
# Normal audio + normal reward
sessionType = clh.SessionType.normal_audio_normal_reward

# No audio + random reward (control)
sessionType = clh.SessionType.no_audio_random_reward
```

## Adaptive Thresholding

When `adaptive_threshold = 1`:

- Every 30 seconds, check rewards in epoch
- If > 1 reward: increase threshold by 0.002
- If 0 rewards: decrease threshold by 0.002

This maintains consistent reward rates across sessions.

## Hardware Setup

### Required Components

- Raspberry Pi 4B+
- Raspberry Pi Camera (v2)
- MPR121 capacitive touch sensor (lick detection)
- LED array (behavior, reward, fail indicators)
- Water reward valve
- Audio speaker

### GPIO Pinout

| GPIO | Function |
|------|----------|
| 17 | Behavior LED |
| 27 | Reward LED |
| 12 | Fail LED |
| 21 | Light TTL |

### I2C Connections

- MPR121 on I2C bus (SCL/SDA)

## Data Output

### VideoTimestamp.txt Format

```
frame	time	roi_dff	freq	rew_threshold	reward	trial	audio	lick
```

### Summary CSV

```
mouse_id, session, data_path, start, end, duration, fps, dff_history, n_tones, reward_threshold, rewards, audio, session_type
```

## Example Workflow

1. Edit `config.ini` with your parameters
2. Run: `python brain/cla_reward_punish_1roi.py`
3. Enter mouse ID when prompted
4. Adjust ROI positions in preview window
5. Press `Esc` to start session
6. Monitor real-time feedback
7. Press `Esc` to end or wait for completion

## Troubleshooting

See [Troubleshooting Guide](troubleshooting.md) for common issues.