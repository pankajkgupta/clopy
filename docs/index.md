# CLoPy - Closed-Loop Feedback Training System

CLoPy is a closed-loop feedback training system for neurofeedback and specified movement feedback in mice. This system enables real-time manipulation of neural activity or behavior based on calcium imaging or pose estimation.

!!! info "Paper"
    This work accompanies the research paper: [https://elifesciences.org/reviewed-preprints/105070](https://elifesciences.org/reviewed-preprints/105070)

## Overview

CLoPy implements two complementary closed-loop paradigms:

| Feature | CLNF | CLMF |
|---------|------|------|
| **Full Name** | Closed-Loop Neurofeedback | Closed-Loop Movement Feedback |
| **Input** | Calcium imaging (Gcamp6f) | DeepLabCut pose estimation |
| **Platform** | Raspberry Pi 4B+ | Nvidia Jetson Orin |
| **Feedback** | Audio tones mapped to neural activity | Audio tones mapped to movement speed |
| **Reward** | Water reward on threshold crossing | Water reward on target movement |

## System Architecture

![CLoPy platform](./assets/fig1.png)

## Key Features

- **Real-time Processing**: Sub-second latency for feedback delivery
- **Adaptive Thresholding**: Automatically adjusts reward thresholds based on performance
- **Multiple ROI Support**: Single ROI (1ROI) or dual ROI (2ROI) experiments
- **Audio Feedback**: Sonification of neural activity or movement speed using variable frequency tones
- **Trial-based Structure**: Well-defined trials with rest periods and success/fail outcomes
- **Data Logging**: Comprehensive logging of all parameters and events

## Folder Structure

```
clopy/
│
├── 📁 analysis/                           Data analysis and visualization
│   ├── get_clmf_data.py                  Load and process CLMF behavioral data
│   ├── plot_clmf.py                      Generate CLMF figures and plots
│   └── plot_clnf.py                      Generate CLNF figures and plots
│
├── 📁 behavior/                           CLMF (Movement Feedback) Experiments
│   └── cla_dlc_trials_speed.py           Main CLMF experiment script
│
├── 📁 brain/                              CLNF (Neurofeedback) Experiments
│   ├── cla_reward_punish_1roi.py         CLNF single ROI experiment
│   └── cla_reward_punish_2roi.py         CLNF dual ROI experiment
│
├── 📁 3D-print/                           3D-printable hardware components
│   ├── MirrorHolderFront.dxf
│   ├── MirrorHolderRear.dxf
│   ├── MouseEnclosureTop.dxf
│   ├── MouseHeadFixBack.dxf
│   ├── MouseHeadFixBottom.dxf
│   ├── MouseHeadFixPost.dxf
│   └── MouseHeadFixTop.dxf
│
├── 📁 processed_data/                     Preprocessed data for figure reproduction
│   ├── clmf_kld_df.csv
│   ├── clmf_sessions_df.csv
│   ├── clmf_trials_df.csv
│   └── clnf_sessions_df.csv
│
├── 📁 docs/                               ReadTheDocs documentation
│   ├── 📁 assets/                        Images, animations, and documentation
│   │   ├── fig1.png                      System overview diagram
│   │   ├── fig2.png                      Module architecture diagram
│   │   ├── animation1_clnf.gif            Neurofeedback trial example
│   │   └── *.pdf                         Hardware assembly instructions
│   │
│   ├── mkdocs.yml                        Documentation configuration
│   ├── index.md                          Home page
│   ├── installation.md                   Installation guide
│   ├── quickstart.md                     Quick start tutorial
│   ├── clnf.md                           CLNF experiment guide
│   ├── clmf.md                           CLMF experiment guide
│   ├── configuration.md                  Configuration reference
│   ├── hardware.md                       Hardware setup guide
│   ├── analysis.md                       Data analysis guide
│   ├── troubleshooting.md                Troubleshooting & FAQ
│   ├── api.md                            API reference
│   └── requirements.txt                  Documentation dependencies
│
├── 📄 CameraFactory.py                    Camera abstraction layer
├── 📄 config.ini                          Configuration file for all experiments
├── 📄 helper.py                           Utility functions and enums
├── 📄 roi_manager.py                      ROI definition and management
├── 📄 PiCameraStream.py                   Raspberry Pi camera driver
├── 📄 SentechCameraStream.py              Sentech USB camera driver
├── 📄 VideoStream.py                      Generic video stream interface
├── 📄 README.md                           Project overview
├── 📄 .readthedocs.yaml                   ReadTheDocs configuration
└── 📄 .gitignore                          Git ignore rules
```

### Directory Descriptions

| Directory | Purpose |
|-----------|---------|
| `analysis/` | Scripts for data analysis and figure generation |
| `behavior/` | CLMF experiment scripts (Jetson Orin) |
| `brain/` | CLNF experiment scripts (Raspberry Pi) |
| `3D-print/` | 3D-printable hardware components (DXF format) |
| `processed_data/` | Pre-processed data for recreating paper figures |
| `docs/` | ReadTheDocs documentation source files and assets |
| `docs/assets/` | Images, animations, and visual files |

## Quick Links

- [Installation Guide](installation.md)
- [CLNF Experiment Setup](clnf.md)
- [CLMF Experiment Setup](clmf.md)
- [Configuration Reference](configuration.md)
- [Hardware Assembly](hardware.md)
- [Troubleshooting](troubleshooting.md)