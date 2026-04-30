# Data Analysis

CLoPy includes scripts for analyzing experimental data and recreating figures from the paper.

## Analysis Scripts

### Location

All analysis scripts are in the `analysis/` folder:

```
analysis/
├── get_clmf_data.py    # Load and process CLMF data
├── plot_clmf.py        # Generate CLMF figures
└── plot_clnf.py        # Generate CLNF figures
```

## CLNF Analysis

### Loading Data

```python
import helper as clh

# Load configuration from session
cfgDict = clh.GetConfigValues('/path/to/session/')
```

### Processing ΔF/F

The system calculates ΔF/F as:

```python
dff = (image - runningAvg) / runningAvg
dffCorrected = dffGreen - dffBlue  # Ratiometric correction
```

### Plotting Functions

```python
# Plot ROI activity over time
clh.plt_roi_activity(mouse_id, data_dir)

# Plot threshold crossings
clh.plt_threshold_crossings(mouse_id, data_dir)

# Plot spontaneous thresholds
clh.plt_spontaneous_thresholds(mouse_id, data_dir)
```

### Data Files

- `VideoTimestamp.txt`: Trial-by-trial log
- `image_stream.hdf5`: Raw imaging data
- `config.ini`: Session configuration

### VideoTimestamp.txt Format (CLNF)

```
frame	time	roi_dff	freq	rew_threshold	reward	trial	audio	lick
```

## CLMF Analysis

### Loading Pose Data

```python
import pandas as pd

# Load trial log
df = pd.read_csv('/path/to/session/log.txt', sep='\t')

# Access pose data
pose_cols = [col for col in df.columns if '_x' in col or '_y' in col]
```

### Calculating Speed

```python
import math

# Calculate speed between frames
speed = math.dist(prev_pos, curr_pos)
```

### Plotting Functions

```python
# In plot_clmf.py
import matplotlib.pyplot as plt

# Plot trial outcomes
plt.figure()
plt.plot(df['trial'], df['reward'])

# Plot speed over time
plt.figure()
plt.plot(df['cj_speed'])
```

### VideoTimestamp.txt Format (CLMF)

```
frame	time	cj_speed	freq	reward	trial	audio	lick	pose_x	pose_y	likelihood...
```

## Preprocessed Data

Place CSV files in `processed_data/` to recreate figures:

```
processed_data/
├── clmf_kld_df.csv
├── clmf_sessions_df.csv
├── clmf_trials_df.csv
└── clnf_sessions_df.csv
```

## Example: Plot Session Summary

```python
import pandas as pd
import matplotlib.pyplot as plt

# Load summary data
df = pd.read_csv('processed_data/clnf_sessions_df.csv')

# Plot reward rate over sessions
plt.figure(figsize=(10, 6))
for mouse in df['mouse_id'].unique():
    mouse_data = df[df['mouse_id'] == mouse]
    plt.plot(mouse_data['session'], mouse_data['rewards'], label=mouse)

plt.xlabel('Session')
plt.ylabel('Total Rewards')
plt.legend()
plt.title('Learning Curve')
plt.show()
```

## Statistical Analysis

### Reward Rate Calculation

```python
# Calculate rewards per minute
sess_dur_min = duration / 60
reward_rate = total_rewards / sess_dur_min
```

### Threshold Optimization

```python
# Find threshold for target reward rate
def find_threshold(dff_values, target_rate):
    for thresh in np.arange(0, 0.2, 0.001):
        rate = sum(dff_values > thresh) / sess_dur_min
        if rate <= target_rate:
            return thresh
    return None
```

## Exporting Data

### To CSV

```python
df.to_csv('output.csv', index=False)
```

### To HDF5

```python
import tables

with tables.open_file('output.hdf5', mode='w') as f:
    f.create_earray(f.root, 'data', obj=array)
```

## Creating Figures

See the paper for figure recreation scripts. Key plots include:

1. **Reward-centered dorsal maps**: Average activity around reward times
2. **Learning curves**: Rewards per session over training
3. **Trial timing**: Distribution of trial durations
4. **Threshold distributions**: Adaptive threshold values