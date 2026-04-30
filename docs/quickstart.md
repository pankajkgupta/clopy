# Quick Start Guide

Get up and running with CLoPy in minutes.

## Running Your First Experiment

### CLNF Experiment (Single ROI)

```bash
cd clopy
python brain/cla_reward_punish_1roi.py
```

### CLNF Experiment (Dual ROI)

```bash
cd clopy
python brain/cla_reward_punish_2roi.py
```

### CLMF Experiment

```bash
cd clopy
python behavior/cla_dlc_trials_speed.py
```

## What Happens Next

1. **Mouse ID Prompt**: Enter your mouse identifier
   ```
   Please enter mouse ID: M001
   ```

2. **Preview Window**: A preview window appears showing:
   - Live camera feed
   - ROI positions (draggable rectangles)
   - DFF (ΔF/F) visualization
   - Luminosity sparklines

3. **Verify Setup**: Ensure:
   - Camera is properly positioned
   - ROI covers the target brain region
   - Bregma landmark is visible
   - Lighting is adequate

4. **Start Session**: Press `Esc` to begin the experiment

5. **During Experiment**:
   - Trials start automatically after initial rest period
   - Audio tones play during active trials
   - Green LED flashes on successful trials
   - Red LED flashes on failed trials

6. **End Session**: Press `Esc` to stop early, or let it complete

## Understanding the Trial Flow

```
┌─────────────────────────────────────────────────────────────┐
│                      Trial Structure                         │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌──────────┐    ┌──────────────┐    ┌────────────┐        │
│  │  Rest    │───▶│    Trial     │───▶│  Outcome   │        │
│  │ Period   │    │   (max 30s)  │    │            │        │
│  └──────────┘    └──────────────┘    └────────────┘        │
│       │                                    │                │
│       ▼                                    ▼                │
│  10-30 seconds                    ┌───────┴───────┐        │
│                                   │               │        │
│                              Success            Fail        │
│                              (reward)         (buzzer)      │
│                                   │               │        │
│                                   ▼               ▼        │
│                              10s rest        15s rest       │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

## Key Controls

| Key | Action |
|-----|--------|
| `Esc` | Start/Stop experiment |
| Mouse Drag | Move ROI positions |
| `Esc` (during trial) | Abort session |

## Data Output

After each session, data is saved to:

```
data_root/mouse_id/YYYYMMDDHHMMSS/
├── config.ini              # Session configuration
├── image_stream.hdf5       # Raw imaging data (CLNF)
├── video.mp4               # Behavior video (CLMF)
└── VideoTimestamp.txt      # Trial-by-trial log
```

## Configuration First

Before running experiments, you'll need to configure:

1. **Camera settings** in `config.ini`
2. **ROI positions** (brain regions)
3. **Reward thresholds**
4. **Trial parameters**

See [Configuration Guide](configuration.md) for details.