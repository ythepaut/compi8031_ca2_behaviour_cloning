# Behvious cloning for a self-driving car simulation

## About

Project carried out as part of a CA of the Smart Technologies (COMPI8031) module at Dundalk Institute of Technology.


## Installation

`pip install -r requirements.txt`


## Datasets

| Dataset folder       | Description                                                               |
|----------------------|---------------------------------------------------------------------------|
| track1_3laps         | Keyboard-controlled, 3 laps in each direction in track 1                  |
| track1_5laps         | Keyboard-controlled, 5 laps in each direction in track 1                  |
| track1_20laps_smooth | Mouse-controlled, 20 laps in each direction                               |
| track2_shadow_part   | Keyboard-controlled, very small part of the track where there is a shadow |
| track2               | Keyboard-controlled, 3 laps in each direction                             |


## Trained models

In tested order :

| Model                                   | Colour representation | Datasets                                                 | Shadow augmentation   | Other                                                                                | Result on track 1                                       | Result on track 2                                                                |
|-----------------------------------------|-----------------------|----------------------------------------------------------|-----------------------|--------------------------------------------------------------------------------------|---------------------------------------------------------|----------------------------------------------------------------------------------|
| model_yuv                               | YUV                   | track1_5laps                                             | No                    |                                                                                      | Drives fine, completes multiple laps                    | Drives on the right and immediately crashes                                      |
| model_rgb                               | RGB                   | track1_5laps                                             | No                    |                                                                                      | Drives fine, completes multiple laps                    | Drives on the right and immediately crashes                                      |
| model_hsv                               | HSV                   | track1_5laps                                             | No                    |                                                                                      | Drives fine, completes multiple laps                    | Manage to drive on the road for the first seconds then crashes on the first turn |
| model_hsv_shadow                        | HSV                   | track1_5laps                                             | Yes (2944bf9 version) |                                                                                      | Drives fine, completes multiple laps                    | Drives fine until the road gets into a shadow, then the car crashes on the left  |
| model_hsv_large_shadow                  | HSV                   | track1_3laps, track1_5laps, track1_20laps_smooth         | Yes                   |                                                                                      | Drives perfectly, completes multiple laps at full speed | Drives fine until the road gets into a shadow, then the car crashes on the left  |
| model_hsv_track1-20_track2-part         | HSV                   | track1_20laps_smooth, track2_shadow_part                 | No                    |                                                                                      | Exits the road just before the bridge                   | Crashes inside the shadow part                                                   |
| model_hsv_large_track2                  | HSV                   | track1_3laps, track1_5laps, track1_20laps_smooth, track2 | No                    |                                                                                      | Exits the road immediately                              | Nearly crashes entering a shadow, crashes exiting it                             |
| model_hsv_track1-20_shadow-v2           | HSV                   | track1_20laps_smooth                                     | Yes (f1e397d version) |                                                                                      | Drives perfectly, completes multiple laps at full speed | Crashes immediately on the signs on the left side                                |
| model_hsv_track1-20_shadow-v2_7000-bins | HSV                   | track1_20laps_smooth                                     | Yes (f1e397d version) | With 7000 samples per bin to add bias so that the car drives more on a straight line | Drives perfectly, completes multiple laps at full speed | Crashes immediately on the signs on the left side                                |
| model_hsv_track1-20_track2              | HSV                   | track1_20laps_smooth, track2                             | Yes (f1e397d version) |                                                                                      | Exits the road immediately                              | Drives okay (20mph limit), then hits an obstacle just outside the road           |
| model_hsv_track2                        | HSV                   | track2                                                   | No                    |                                                                                      | Exits the road just before the bridge                   | Drives okay (20mph limit), then hits an obstacle just outside the road           |
| model_hsv_track1-20_50epochs            | HSV                   | track1_20laps_smooth                                     | Yes (f1e397d version) | With 50 epochs                                                                       | Drives perfectly, completes multiple laps at full speed | Manage to drive on the road for the first seconds then crashes on the first turn |
| model_hsv_track1-20_track2_50epochs     | HSV                   | track1_20laps_smooth, track2                             | Yes (f1e397d version) | With 50 epochs                                                                       | Exits the road just before the bridge                   | Drives okay (20mph limit), completes multiple laps                               |

Best model for track 1 : `model_hsv_large_shadow.h5`, maximum speed.

Best model for track 2 : `model_hsv_track1-20_track2_50epochs.h5`, 20mph speed limit.

## Authors

[@Zarvoira](https://github.com/Zarvoira) - Agung Santosa

[@ythepaut](https://github.com/ythepaut) - Yohann Thepaut
