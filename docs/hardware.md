# Hardware Setup

Detailed hardware assembly instructions for CLoPy experiments.

!!! info "Parts List"
    Complete parts lists are available in the PDF: [CLNF_Parts_List_and_Assembly_Instructions_Gupta_et_al.pdf](https://github.com/pankajkgupta/clopy/blob/main/assets/CLNF_Parts_List_and_Assembly_Instructions_Gupta_et_al.pdf)

## CLNF Hardware (Raspberry Pi)

### Components

| Component | Quantity | Notes |
|-----------|----------|-------|
| Raspberry Pi 4B+ | 1 | |
| Raspberry Pi Camera v2 | 1 | |
| MPR121 Capacitive Touch | 1 | Lick detection |
| LEDs (various) | 4 | Behavior, Reward, Fail, TTL |
| Water valve | 1 | Reward delivery |
| Speaker | 1 | Audio feedback |
| Power supply | 1 | 5V 3A minimum |

### Pin Connections

#### GPIO Pinout (Board numbering)

```
┌─────────────────────────────────────────────┐
│              Raspberry Pi GPIO              │
├─────────────────────────────────────────────┤
│                                             │
│  3.3V  ──── VCC (MPR121)                    │
│  GND   ──── GND (MPR121, LEDs)              │
│  SCL   ──── SCL (MPR121)                    │
│  SDA   ──── SDA (MPR121)                    │
│                                             │
│  GPIO17 ──── Behavior LED (pin 1)           │
│  GPIO27 ──── Reward LED (pin 1)             │
│  GPIO12 ──── Fail LED (pin 1)               │
│  GPIO21 ──── Light TTL (pin 1)              │
│                                             │
└─────────────────────────────────────────────┘
```

#### LED Wiring

```
GPIO ──── Resistor (330Ω) ──── LED ──── GND
```

#### Water Valve

Connect to relay controlled by Arduino or GPIO.

### Camera Setup

1. Connect camera to CSI port
2. Enable camera in `raspi-config`:
    ```bash
    sudo raspi-config
    ```
    → Interface Options → Camera → Enable

3. Test camera:
    ```bash
    raspistill -o test.jpg
    ```

### I2C Setup for MPR121

1. Enable I2C:
    ```bash
    sudo raspi-config
    ```
    → Interface Options → I2C → Enable

2. Install tools:
    ```bash
    sudo apt-get install i2c-tools
    ```

3. Verify connection:
    ```bash
    sudo i2cdetect -y 1
    ```
    Should show device at address `0x5a`

## CLMF Hardware (Jetson Orin)

### Components

| Component | Quantity | Notes |
|-----------|----------|-------|
| Nvidia Jetson Orin | 1 | |
| Sentech Camera | 1 | USB3 preferred |
| Arduino Uno/Mega | 1 | TTL and LED control |
| Water valve | 1 | Reward delivery |
| Speaker | 1 | Audio feedback |

### Arduino Connections

#### Pinout

```
┌─────────────────────────────────────────────┐
│                 Arduino                     │
├─────────────────────────────────────────────┤
│                                             │
│  13  ──── Brain TTL (output)                │
│  7   ──── Reward LED                        │
│  12  ──── Fail LED                          │
│  40  ──── Light TTL                         │
│                                             │
│  GND  ──── GND (LEDs, TTL)                  │
│                                             │
│  Serial: /dev/ttyACM0                       │
│                                             │
└─────────────────────────────────────────────┘
```

#### Python Serial Configuration

```python
board = Arduino("115200", port="/dev/ttyACM0")
```

!!! note "Serial Port"
    The serial port may vary. Check with:
        ```bash
        ls /dev/ttyACM*
        ```

### Camera Setup (Sentech)

1. Connect via USB3
2. Verify:
   ```bash
   v4l2-ctl --list-devices
   ```

3. Test capture:
   ```bash
   gst-launch-1.0 v4l2src ! videoconvert ! ximagesink
   ```

### GPIO (Jetson)

Using RPi.GPIO library for compatibility:

```python
import RPi.GPIO as GPIO

GPIO.setmode(GPIO.BOARD)
GPIO.setup(pin, GPIO.OUT, initial=GPIO.LOW)
```

## Optical Setup

### Microscopy Configuration

```
                    ┌──────────────┐
                    │   LED Array  │
                    │  (470nm for  │
                    │   GCaMP6f)   │
                    └──────┬───────┘
                           │
                           ▼
              ┌────────────────────────┐
              │    Filter Set          │
              │  (Excitation/Emission) │
              └───────────┬────────────┘
                          │
                          ▼
              ┌────────────────────────┐
              │    Camera              │
              │  (Pi Camera / Sentech) │
              └────────────────────────┘
```

### Illumination

- **CLNF**: 470nm LED for GCaMP6f excitation
- Use appropriate filter sets for fluorescence imaging

### Calibration

1. **ppmm (Pixels per mm)**:
   - Image a ruler at your optical magnification
   - Count pixels across known distance
   - Set in config: `ppmm = <count>`

2. **Bregma Location**:
   - Image the skull landmarks
   - Note the (row, col) coordinates
   - Set in config: `bregma = <row>, <col>`

## Mouse Headfixing

### Setup

1. Headfix the mouse using the 3D-printed parts:
   - `MouseHeadFixTop.dxf`
   - `MouseHeadFixBottom.dxf`
   - `MouseHeadFixPost.dxf`

2. Position under camera with:
   - Cortex visible
   - Bregma identifiable
   - Target region in field of view

### ROI Placement

1. During preview, drag ROIs to:
   - Target brain region
   - Reference region (for dual ROI)

2. Positions are saved to `config.ini` after preview

## Audio Setup

### Speaker Requirements

- Frequency response: 1-24 kHz
- Small form factor for setup near mouse

### Audiostream Configuration

```python
from audiostream import get_output
from audiostream.sources.wave import SineSource

austream = get_output(channels=1, rate=44100, buffersize=128)
sinsource = SineSource(austream, 1000)
sinsource.start()
```

## Power Considerations

### Raspberry Pi

- Recommended: 5V 3A power supply
- Can power LEDs directly (with resistors)
- Water valve may need external power

### Jetson Orin

- Use official power supply
- Ensure adequate cooling
- Monitor power consumption for GPU work

## Safety Notes

!!! warning
- Always use appropriate resistors for LEDs
- Verify polarity before connecting
- Keep water valve wiring separate from logic
- Ensure proper grounding