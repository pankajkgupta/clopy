# CLoPy
Closed-loop feedback training system (CLoPy) for neurofeedback and specified movement feedback in mice

![CLoPy](https://github.com/pankajkgupta/clopy/blob/main/assets/fig1.png?raw=true)

```
clopy/
├── analysis/
│   ├── get_clmf_data.py
│   ├── get_clnf_data.py
│   ├── plot_clmf.py
│   └── plot_clnf.py
├── assets/
│   ├── fig1.png
│   ├── fig2.png
│   ├── GT33_tta_20230728121232_rewbehbrain20594.gif
│   ├── GT33_tta_20231004180719_rewbehbrain9898.gif
│   └── supplementary5.png
├── behavior/
│   └── cla_dlc_trials_speed.py
├── brain/
│   ├── cla_reward_punish_1roi.py
│   └── cla_reward_punish_2roi.py
├── CameraFactory.py
├── config.ini
├── helper.py
├── PiCameraStream.py
├── processed_data/
├── README.md
├── roi_manager.py
├── SentechCameraStream.py
└── VideoStream.py
```

![CLoPy](https://github.com/pankajkgupta/clopy/blob/main/assets/fig2.png?raw=true)

Click this to watch an overview

[![CLoPy](<image path>)](<youtube URL>)

![CLoPy](https://github.com/pankajkgupta/clopy/blob/main/assets/GT33_tta_20230728121232_rewbehbrain20594.gif?raw=true)
![CLoPy](https://github.com/pankajkgupta/clopy/blob/main/assets/GT33_tta_20231004180719_rewbehbrain9898.gif?raw=true)


How-to
---------------------------

1. Install Python 3.x (anaconda recommended)

2. Install PyAudio

sudo apt-get install portaudio19-dev python3-pyaudio

pip3 install PyAudio

3. # Install audiostream
sudo apt-get install libsdl1.2-dev

sudo apt-get install libsdl-mixer1.2-dev

pip3 install cython

pip3 install kivy

4. Clone the repository:

   ```bash
   $ git clone <>
   ```

   or download as zip and extract.

5. In the root directory run

   ```bash
   $ python <>
   ```

6. Use <kbd>Esc</kbd> to close the program.

