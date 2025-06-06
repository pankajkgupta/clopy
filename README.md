# CLoPy
Closed-loop feedback training system (CLoPy) for neurofeedback and specified movement feedback in mice

This work accompanies the research paper - 
https://www.biorxiv.org/content/10.1101/2024.11.02.619716v1

## Overall modules in the system:

![CLoPy](https://github.com/pankajkgupta/clopy/blob/main/assets/fig1.png?raw=true)


## Folder structure
```
clopy/
├── analysis/                                               Folder with analysis code
│   ├── get_clmf_data.py
│   ├── plot_clmf.py
│   └── plot_clnf.py
├── assets/                                                 Folder with images and animations for display
│   ├──
│   . 
│   . 
│   . 
│    
├── behavior/                                               Folder with script to run CLMF experiment
│   └── cla_dlc_trials_speed.py
├── brain/                                                  Folder with scripts to run CLNF experiment
│   ├── cla_reward_punish_1roi.py
│   └── cla_reward_punish_2roi.py
├── CameraFactory.py
├── config.ini
├── helper.py
├── PiCameraStream.py
├── processed_data/                                         Place preprocessed data in this folder to recreate figures
├── README.md
├── roi_manager.py
├── SentechCameraStream.py
└── VideoStream.py
```


# Setup

Overview of CLNF process (A) and CLMF process (B)

![CLoPy](https://github.com/pankajkgupta/clopy/blob/main/assets/fig2.png?raw=true)



## CLNF setup

> [!NOTE]
> CLNF was implemented on Raspberry Pi 4B+ and the steps below are to replicate that. 
> But the system can be adapted to other platforms with minor adaptations.

The rig hardware parts list can be found here- [CLNF_Parts_List_and_Assembly_Instructions_Gupta_et_al.pdf](https://github.com/pankajkgupta/clopy/blob/main/assets/CLNF_Parts_List_and_Assembly_Instructions_Gupta_et_al.pdf)

How-to
---------------------------

Install Python >= 3.8 ([anaconda](https://docs.anaconda.com/anaconda/install/) or [miniconda](https://docs.anaconda.com/miniconda/miniconda-install/) recommended)

> [!TIP]
> We highly recommend creating a virtual environment such as using conda and installing all the Python packages in that environment

The following commands are to be run in a Terminal (Linux/MacOS)

```
conda create -n clopy
```
Activate the conda environment

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
pip3 install cython
```
```
pip3 install kivy
```

Clone the Audiostream repo:
```
git clone https://github.com/kivy/audiostream.git
```
or download as a zip and extract.

Change the directory to the downloaded repository

```
cd audiostream
```
Run Audiostream setup installation
```
sudo python setup.py install
```

After setup completes successfully, change the current directory to a directory where you want to clone CLoPy (eg. home directory)
```
cd ~
```

Clone the CLoPy repository:

```bash
git clone https://github.com/pankajkgupta/clopy.git
```

   or download as a zip and extract.

```
cd clopy
```

In the clopy root directory run

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

#### Here are some reward-centered dorsal cortical maps averaged over trials of a session:

![CLoPy](https://github.com/pankajkgupta/clopy/blob/main/assets/animation1_clnf.gif?raw=true)

## CLMF setup

> [!NOTE]
> CLMF was implemented on Nvidia-Jetson Orin and steps below are to replicate that. 
> But the system can be adapted to other platforms with a GPU capable of inference.

The rig hardware parts list can be found here- [CLMF_Parts_List_and_Assembly_Instructions_Gupta_et_al.pdf](https://github.com/pankajkgupta/clopy/blob/main/assets/CLMF_Parts_List_and_Assembly_Instructions_Gupta_et_al.pdf)

How-to
---------------------------

Install Python >= 3.8 ([anaconda](https://docs.anaconda.com/anaconda/install/) or [miniconda](https://docs.anaconda.com/miniconda/miniconda-install/) recommended)

> [!TIP]
> We highly recommend creating a virtual environment such as using conda and installing all the Python packages in that environment

```
conda create -n clopy
```
Activate the conda environment

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
pip3 install cython
```
```
pip3 install kivy
```

Clone the Audiostream repo:
```
git clone https://github.com/kivy/audiostream.git
```
or download as zip and extract.

Change the directory to the downloaded repository

```
cd audiostream
```
Run Audiostream setup installation
```
sudo python setup.py install
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

After setup completes successfully, change the current directory to a directory where you want to clone CLoPy (eg. home directory)
```
cd ~
```

Clone the CLoPy repository:

```bash
git clone https://github.com/pankajkgupta/clopy.git
```

   or download as zip and extract.

```
cd clopy
```

In the clopy root directory run

```bash
python behavior/cla_dlc_trials_speed.py
```

... you would be prompted to enter a 'mouse_id'. After this step, you will see a 
preview window where you can make sure all settings and setup look fine 
before hitting <kbd>Esc</kbd> button the the keyboard to start a session.

If you want to interrupt the session after it starts, press the <kbd>Esc</kbd> button while the window is selected, to close the session as the program.


#### Here are two example trials from the CLMF experiment where the task was to move the left fore-limb (top animation)

![CLoPy](https://github.com/pankajkgupta/clopy/blob/main/assets/GT33_tta_20230728121232_rewbehbrain20594.gif?raw=true)

#### ... and later the same mouse was trained to move the right fore-limb

![CLoPy](https://github.com/pankajkgupta/clopy/blob/main/assets/GT33_tta_20231004180719_rewbehbrain9898.gif?raw=true)
