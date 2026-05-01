# CLoPy
Closed-loop feedback training system (CLoPy) for neurofeedback and specified movement feedback in mice

This work accompanies the research paper - 
https://www.biorxiv.org/content/10.1101/2024.11.02.619716v1

## Folder structure
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


## Setup

Overview of CLNF process (A) and CLMF process (B)

![CLoPy](https://github.com/pankajkgupta/clopy/blob/main/docs/assets/fig1.png?raw=true)


## Overall modules in the system:

![CLoPy](https://github.com/pankajkgupta/clopy/blob/main/docs/assets/fig2.png?raw=true)



## CLNF setup

> [!NOTE]
> CLNF was implemented on Raspberry Pi 4B+ and the steps below are to replicate that. 
> But the system can be adapted to other platforms with minor adaptations.

The rig hardware parts list can be found here- [CLNF_Parts_List_and_Assembly_Instructions_Gupta_et_al.pdf](https://github.com/pankajkgupta/clopy/blob/main/docs/assets/CLNF_Parts_List_and_Assembly_Instructions_Gupta_et_al.pdf)

How-to
---------------------------

Install Python >= 3.8 ([anaconda](https://docs.anaconda.com/anaconda/install/) or [miniconda](https://docs.anaconda.com/miniconda/miniconda-install/) recommended)

> [!TIP]
> We highly recommend creating a virtual environment such as using conda and installing all the Python packages in that environment

The following commands are to be run in a Terminal (Linux/MacOS)

```
conda create -n clopy
```
### Activate the conda environment

```
conda activate clopy
```

### Install dependencies
```
pip3 install opencv-python tables cvui roipoly scipy pandas joblib pillow videofig imutils gpiozero Adafruit-Blinka adafruit-circuitpython-mpr121
```

### Install [PyAudio](https://people.csail.mit.edu/hubert/pyaudio/)
Follow your platform-specific instructions. The commands below are specific to Linux-based systems

```
sudo apt-get install portaudio19-dev python3-pyaudio
```
```
pip3 install PyAudio
```

### Install [Audiostream](https://github.com/kivy/audiostream)

```
sudo apt-get install libsdl1.2-dev
```
```
sudo apt-get install libsdl-mixer1.2-dev
```
```
pip3 install cython==0.29.21
```
```
pip3 install kivy
```

### Clone the Audiostream repo:
```
git clone https://github.com/kivy/audiostream.git
```
or download as a zip and extract.

### Change the directory to the downloaded repository

```
cd audiostream
```
### Run Audiostream setup installation
```
python setup.py install
```

### After setup completes successfully, change the current directory to a directory where you want to clone CLoPy (eg. home directory)
```
cd ~
```

### Clone the CLoPy repository:

```bash
git clone https://github.com/pankajkgupta/clopy.git
```

   or download as a zip and extract.

```
cd clopy
```

### In the clopy root directory run

For CLNF experiment involving single ROI (1ROI):
```bash
python brain/cla_reward_punish_1roi.py
```
For CLNF experiment involving dual ROI (2ROI):
```bash
python brain/cla_reward_punish_2roi.py
```

... you would be prompted to enter a 'mouse_id'. After this step, you will see a 
preview window where you can make sure all settings and setup look fine 
before hitting <kbd>Esc</kbd> button the the keyboard to start a session.

If you want to interrupt the session after it starts, press the <kbd>Esc</kbd> button while the window is selected, to close the session as the program.

### Here are some reward-centered dorsal cortical maps averaged over trials of a session:

![CLoPy](https://github.com/pankajkgupta/clopy/blob/main/docs/assets/animation1_clnf.gif?raw=true)

## CLMF setup

> [!NOTE]
> CLMF was implemented on Nvidia-Jetson Orin and steps below are to replicate that. 
> But the system can be adapted to other platforms with a GPU capable of inference.

The rig hardware parts list can be found here- [CLMF_Parts_List_and_Assembly_Instructions_Gupta_et_al.pdf](https://github.com/pankajkgupta/clopy/blob/main/docs/assets/CLMF_Parts_List_and_Assembly_Instructions_Gupta_et_al.pdf)

How-to
---------------------------

Install Python >= 3.8 ([anaconda](https://docs.anaconda.com/anaconda/install/) or [miniconda](https://docs.anaconda.com/miniconda/miniconda-install/) recommended)

> [!TIP]
> We highly recommend creating a virtual environment such as using conda and installing all the Python packages in that environment

```
conda create -n clopy
```
### Activate the conda environment

```
conda activate clopy
```

### Install dependencies
```
pip3 install opencv-python tables cvui roipoly scipy pandas joblib pillow videofig imutils
```
### Install [PyAudio](https://people.csail.mit.edu/hubert/pyaudio/)
Follow your platform-specific instructions. The commands below are specific to Linux-based systems

```
sudo apt-get install portaudio19-dev python3-pyaudio
```
```
pip3 install PyAudio
```

### Install [Audiostream](https://github.com/kivy/audiostream)

```
sudo apt-get install libsdl1.2-dev
```
```
sudo apt-get install libsdl-mixer1.2-dev
```
```
pip3 install cython==0.29.21
```
```
pip3 install kivy
```

### Clone the Audiostream repo:
```
git clone https://github.com/kivy/audiostream.git
```
or download as zip and extract.

### Change the directory to the downloaded repository

```
cd audiostream
```
### Run Audiostream setup installation
```
python setup.py install
```

### Install DeepLabCut-Live
> [!IMPORTANT]
> Please check the DeepLabCut-Live ([Kane et al, eLife 2020](https://elifesciences.org/articles/61909)) [GitHub page](https://github.com/DeepLabCut/DeepLabCut-live) for latest instructions for installing this package. 
> Please also check the tensorflow version requirements on their GitHub page. 
> 
> If all dependencies are fine, the command below usually works fine

```
pip install deeplabcut-live
```

### After setup completes successfully, change the current directory to a directory where you want to clone CLoPy (eg. home directory)
```
cd ~
```

### Clone the CLoPy repository:

```bash
git clone https://github.com/pankajkgupta/clopy.git
```

   or download as zip and extract.

```
cd clopy
```

### In the clopy root directory run

```bash
python behavior/cla_dlc_trials_speed.py
```

... you would be prompted to enter a 'mouse_id'. After this step, you will see a 
preview window where you can make sure all settings and setup look fine 
before hitting <kbd>Esc</kbd> button the the keyboard to start a session.

If you want to interrupt the session after it starts, press the <kbd>Esc</kbd> button while the window is selected, to close the session as the program.


### Here are two example trials from the CLMF experiment where the task was to move the left fore-limb (top animation)

![CLoPy](https://github.com/pankajkgupta/clopy/blob/main/docs/assets/GT33_tta_20230728121232_rewbehbrain20594.gif?raw=true)

### ... and later the same mouse was trained to move the right fore-limb

![CLoPy](https://github.com/pankajkgupta/clopy/blob/main/docs/assets/GT33_tta_20231004180719_rewbehbrain9898.gif?raw=true)
