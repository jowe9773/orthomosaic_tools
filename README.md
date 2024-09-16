# Orthomosaic_Tools
This repository contains programs to orthomosaic videos from 4 overhead cameras. 

## Downloading and Setting Up These Programs


## Selecting Targets
In order to orthorectify the videos on each camera and line them up in space, the x and y pixel coordinates of targets within each cameras frame must be known and related to real world coordinates. Included in this repository is a program to find the pixel locations of targets with an image or video stream (as long and the targets are not moving in relation to the camera). To use it, run the file "*gcp_selector.py*". When you do this, a gui will open up and look like the image below. This window can be resized as needed for your computer. 

![gcp_selector_gui_open](https://github.com/user-attachments/assets/bd62185f-203f-42fc-aa06-11f1561e43b3)

The first step in using this program is to load a video or image of the same camera settings as the videos you are interested in orthorectifying into the program. To do this, click on the **Load Image/Video** button in the top right of the window. Navigate your file explorer in the window that pops up and open the image/video file of your choosing. Ideally, this image/video should be of the flume with no water on top of it so that the position of the targets are not distorted by the refraction caused by water. Once the file is loaded, the the left portion of the window should populate with the image or the first frame of the video. It will look like the image below.

![gcp_selector_gui_image_opened](https://github.com/user-attachments/assets/b5669baf-92da-4993-8ff7-b7f6eadc2298)



## Orthorectifying Videos
