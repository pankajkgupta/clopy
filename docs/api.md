# API Reference

Detailed API documentation for CLoPy modules and classes.

## Core Modules

### CameraFactory

Factory class for creating camera instances.

```python
from CameraFactory import CameraFactory
```

#### Methods

##### `CameraFactory(camera_instance)`

Create a camera wrapper.

**Parameters:**
- `camera_instance`: Camera object from camera driver

**Returns:** Wrapped camera object

**Example:**
```python
from PiCameraStream import PiCameraStream
vs = CameraFactory(PiCameraStream(cfgDict).start())
```

---

### roi_manager

ROI management for defining and manipulating regions of interest.

#### Classes

##### `Rect`

Rectangle ROI class.

```python
from roi_manager import Rect
```

**Attributes:**
- `x`, `y`: Top-left corner coordinates
- `w`, `h`: Width and height
- `name`: ROI identifier
- `color`: BGR color tuple
- `avgDff`: Current ΔF/F value

**Methods:**

```python
roi = Rect('M1_L', x=100, y=100, w=20, h=20, color=[0, 0, 255])
area = roi.area()  # Returns w * h
```

##### `Circle`

Circular ROI class.

```python
from roi_manager import Circle
```

**Attributes:**
- `x`, `y`: Center coordinates
- `r`: Radius
- `name`: ROI identifier
- `color`: BGR color tuple

**Methods:**

```python
roi = Circle('start', x=100, y=100, r=30, color=[0, 200, 0])
area = roi.area()  # Returns π * r²
```

##### `annots`

Annotation container class.

```python
from roi_manager import annots
```

**Attributes:**
- `rois`: Dictionary of ROI objects
- `image`: Current frame
- `frame_n`: Frame number
- `wname`: Window name

---

### helper

Utility functions for CLoPy.

```python
import helper as clh
```

#### Functions

##### `GetConfigValues(data_dir)`

Load configuration from session directory.

```python
cfgDict = clh.GetConfigValues('/path/to/session/')
```

**Parameters:**
- `data_dir`: Path to session directory

**Returns:** Dictionary of configuration values

##### `generate_seeds(bregma, seeds_mm, ppmm, side)`

Generate pixel coordinates for brain regions.

```python
br = clh.Position(135, 128)  # bregma position
seeds = clh.generate_seeds(br, seeds_mm, ppmm, 'u')
```

**Parameters:**
- `bregma`: Position object with row, col
- `seeds_mm`: Dictionary of region coordinates in mm
- `ppmm`: Pixels per millimeter
- `side`: Side indicator ('u' for unilateral)

**Returns:** Dictionary of pixel coordinates

##### `get_freqs(n_tones)`

Generate frequency values for audio feedback.

```python
freqs = clh.get_freqs(18)
```

**Parameters:**
- `n_tones`: Number of frequency bins

**Returns:** Dictionary mapping bin indices to frequencies

#### Enums

##### `SessionType`

Experiment session types.

```python
from helper import SessionType

sessionType = SessionType.normal_audio_normal_reward
```

**Values:**
- `normal_audio_normal_reward`: Standard feedback and rewards
- `single_audio_normal_reward`: Constant tone with rewards
- `no_audio_random_reward`: No feedback, random rewards (control)
- `normal_audio_no_reward`: Feedback only, no rewards
- `no_audio_no_reward`: No feedback, no rewards (control)

---

### configparser

Configuration file handling.

```python
from configparser import ConfigParser
```

**Usage:**

```python
config = ConfigParser()
config.read('config.ini')

# Read values
value = config.get('section', 'key')
value = config.getint('section', 'key')
value = config.getfloat('section', 'key')
value = config.getboolean('section', 'key')

# Write values
config.set('section', 'key', 'value')

# Save
with open('config.ini', 'w') as f:
    config.write(f)
```

---

## CLNF Modules

### cla_reward_punish_1roi

Single ROI closed-loop neurofeedback experiment.

```bash
python brain/cla_reward_punish_1roi.py
```

#### Key Variables

| Variable | Type | Description |
|----------|------|-------------|
| `rois` | list | List of Rect objects |
| `runningImgQuROI` | list | Deques for DFF history |
| `relAvgDff` | float | Current ΔF/F value |
| `reward_threshold` | float | Threshold for reward |
| `dff_bins` | ndarray | DFF to frequency mapping |

#### Trial Flow

1. Rest period (configurable duration)
2. Trial starts → audio plays
3. Monitor ΔF/F in ROI
4. If ΔF/F > threshold → reward + success LED
5. If timeout → failure LED
6. Rest period (varies by outcome)

---

### cla_reward_punish_2roi

Dual ROI closed-loop neurofeedback experiment.

```bash
python brain/cla_reward_punish_2roi.py
```

#### Key Differences from 1ROI

- Uses `roi_operation` for mathematical combination
- Tracks two ROIs
- Calculates: `ROI1 + ROI2` or `ROI1 - ROI2`

**Example:**
```python
roi_operation = 'BC_L-BC_R'  # Left - Right barrel cortex
relAvgDff = rois[0].avgDff - rois[1].avgDff
```

---

## CLMF Modules

### cla_dlc_trials_speed

Closed-loop movement feedback experiment.

```bash
python behavior/cla_dlc_trials_speed.py
```

#### Key Variables

| Variable | Type | Description |
|----------|------|-------------|
| `pose` | ndarray | Current pose estimates |
| `cj_speed` | float | Control joint speed |
| `speed_threshold` | int | Speed threshold for reward |
| `control_point_ix` | int | Index of tracked body part |
| `start_roi_manager` | Circle | Start region |
| `target_roi_manager` | Circle | Target region |

#### DeepLabCut-Live Integration

```python
from dlclive import DLCLive, Processor

# Initialize
dlc_proc = Processor()
dlc_live = DLCLive(model_path, processor=dlc_proc, display=False)

# Get pose
pose = dlc_live.get_pose(image)

# Access specific body part
x, y = pose[control_point_ix][0:2]
```

#### Trial Initialization Options

Three modes for starting trials (see code comments):

1. **Stability-based** (default): Paw must be stationary
2. **Start ROI-based**: Paw must be in start circle
3. **Immediate**: Trial starts after rest period

---

## Data Structures

### VideoTimestamp.txt (CLNF)

Tab-separated log file:

```
frame	time	roi_dff	freq	rew_threshold	reward	trial	audio	lick
```

| Column | Type | Description |
|--------|------|-------------|
| frame | int | Frame number |
| time | str | Timestamp (YYYY-MM-DD HH:MM:SS.ffffff) |
| roi_dff | float | ΔF/F value |
| freq | int | Audio frequency (Hz) |
| rew_threshold | float | Current threshold |
| reward | int | -1=fail, 0=none, 1=success |
| trial | int | Trial number |
| audio | int | 0=no audio, 1=audio playing |
| lick | int | 0=no lick, 1=lick detected |

### VideoTimestamp.txt (CLMF)

Tab-separated log file:

```
frame	time	cj_speed	freq	reward	trial	audio	lick	pose_x	pose_y	likelihood...
```

| Column | Type | Description |
|--------|------|-------------|
| cj_speed | float | Control joint speed (pixels/frame) |
| pose_* | float | x, y, likelihood for each body part |

### image_stream.hdf5

HDF5 file containing raw imaging data:

```python
import tables

with tables.open_file('image_stream.hdf5', mode='r') as f:
    images = f.root.raw_images[:]  # Shape: (n_frames, height, width, channels)
```

---

## Configuration Schema

### Required Sections

```ini
[configsection]
config = section_name

[section_name]
# ... parameters
```

### Common Parameters

See [Configuration Reference](configuration.md) for complete parameter list.