# CLoPy
Closed-loop feedback training system (CLoPy) for neurofeedback and specified movement feedback in mice

## Overall modules in the system:

![CLoPy](https://github.com/pankajkgupta/clopy/blob/main/assets/fig1.png?raw=true)


## Folder stucture
```
clopy/
├── analysis/                                               Folder with all analysis code
│   ├── get_clmf_data.py
│   ├── plot_clmf.py
│   └── plot_clnf.py
├── assets/                                                 Folder with images and animations for display
│   ├── fig1.png
│   ├── fig2.png
│   ├── GT33_tta_20230728121232_rewbehbrain20594.gif
│   ├── GT33_tta_20231004180719_rewbehbrain9898.gif
│   └── supplementary5.png
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

![CLoPy](https://github.com/pankajkgupta/clopy/blob/main/assets/fig2.png?raw=true)



## CLNF setup

> [!NOTE]
> CLNF was implemented on Raspberry Pi 4B+ and steps below are to replicate that. 
> But the system can be adapted to other platforms with minor adaptations.

The rig hardware parts list can be found here- [CLNF_Parts_List_and_Assembly_Instructions_Gupta_et_al.pdf](https://github.com/pankajkgupta/clopy/blob/main/assets/CLNF_Parts_List_and_Assembly_Instructions_Gupta_et_al.pdf)

How-to
---------------------------

1. Install Python 3.x (anaconda recommended)

2. Install PyAudio

sudo apt-get install portaudio19-dev python3-pyaudio

pip3 install PyAudio

### Install audiostream

```
sudo apt-get install libsdl1.2-dev

sudo apt-get install libsdl-mixer1.2-dev

pip3 install cython

pip3 install kivy
```

4. Clone the repository:

   ```bash
   $ git clone https://github.com/pankajkgupta/clopy.git
   ```

   or download as zip and extract.

5. In the clopy root directory run

   ```bash
   $ python <>.py
   ```

6. Use <kbd>Esc</kbd> to close the program.

## CLMF setup

> [!NOTE]
> CLMF was implemented on Nvidia-Jetson Orin and steps below are to replicate that. 
> But the system can be adapted to other platforms with a GPU capable of inference.

The rig hardware parts list can be found here- [CLMF_Parts_List_and_Assembly_Instructions_Gupta_et_al.pdf](https://github.com/pankajkgupta/clopy/blob/main/assets/CLMF_Parts_List_and_Assembly_Instructions_Gupta_et_al.pdf)

> [!IMPORTANT]
> Please check the 
> Please check the eepLabCut-Live GitHub page for latest instructions for installing this package. Command below usually works fine

```
pip install deeplabcut-live
```

![CLoPy](https://github.com/pankajkgupta/clopy/blob/main/assets/GT33_tta_20230728121232_rewbehbrain20594.gif?raw=true)
![CLoPy](https://github.com/pankajkgupta/clopy/blob/main/assets/GT33_tta_20231004180719_rewbehbrain9898.gif?raw=true)
