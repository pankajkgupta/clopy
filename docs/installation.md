# Installation Guide

This guide covers installation of CLoPy on your system. Choose the appropriate section based on your experiment type (CLNF or CLMF).

!!! warning "Platform Requirements"
    - **CLNF**: Raspberry Pi 4B+ (recommended)
    - **CLMF**: Nvidia Jetson Orin with GPU (required for DeepLabCut-Live)

## Prerequisites

- Python >= 3.8
- [Anaconda](https://docs.anaconda.com/anaconda/install/) or [Miniconda](https://docs.anaconda.com/miniconda/miniconda-install/) (recommended)

## Step 1: Create Environment

```bash
conda create -n clopy
conda activate clopy
```

## Step 2: Install Core Dependencies

```bash
pip3 install opencv-python tables cvui roipoly scipy pandas joblib pillow videofig imutils
```

## Step 3: Install Audio Dependencies

### Install PyAudio

#### Linux (Debian/Ubuntu)

```bash
sudo apt-get install portaudio19-dev python3-pyaudio
pip3 install PyAudio
```

#### macOS

```bash
brew install portaudio
pip3 install PyAudio
```

#### Windows

Download PyAudio from [Christophe Gauthier's site](https://people.csail.mit.edu/hubert/pyaudio/) or use:

```bash
pip3 install PyAudio
```

### Install Audiostream

#### Linux (Debian/Ubuntu)

```bash
sudo apt-get install libsdl1.2-dev
sudo apt-get install libsdl-mixer1.2-dev
pip3 install cython==0.29.21
pip3 install kivy
```

#### macOS/Windows

Follow [Kivy installation guide](https://kivy.org/doc/stable/gettingstarted/installation.html)

```bash
# Clone audiostream repository
git clone https://github.com/kivy/audiostream.git
cd audiostream
python setup.py install
```

## Step 4: Install Hardware-Specific Dependencies

### For CLNF (Raspberry Pi)

```bash
pip3 install gpiozero Adafruit-Blinka adafruit-circuitpython-mpr121
```

### For CLMF (Jetson Orin)

```bash
pip install deeplabcut-live
```

!!! important "DeepLabCut-Live"
    Please check the [DeepLabCut-Live GitHub page](https://github.com/DeepLabCut/DeepLabCut-live) for the latest installation instructions and TensorFlow version requirements.

## Step 5: Clone CLoPy Repository

```bash
cd ~
git clone https://github.com/pankajkgupta/clopy.git
cd clopy
```

## Step 6: Configure Hardware

### Raspberry Pi I2C Setup

1. Enable I2C interface:

```bash
sudo raspi-config
```

   Navigate to `Interface Options` → `I2C` → `Enable`

2. Install I2C tools:

```bash
sudo apt-get install i2c-tools
```

### Arduino Setup (for CLMF)

- Connect Arduino via USB
- Note the serial port (typically `/dev/ttyACM0`)

## Verification

Test your installation:

```python
import cv2
import numpy as np
from configparser import ConfigParser

print("Core packages imported successfully!")
```

## Next Steps

- [Quick Start Guide](quickstart.md)
- [CLNF Configuration](clnf.md)
- [CLMF Configuration](clmf.md)