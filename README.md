# Orthomosaic_Tools
I built this toolbox to orthorectify videos from 4 nadir cameras above the Riverine Basin at the St. Anthony Falls Laboratory in order to create videos that covered the entire spatial extent of my experiments. Using this program, there are two parts in orthorectifying the video streams. First, extracting the pixel coordinates of targets on an image or video from each camera. This allows us to compute a "homography matrix" which in turn allows us to warp the image to remove lens distortion and camera position effects. Second, using these targets to compute homography matrix, apply the homography matricies to each cameras video stream, and stitching them together into the final product. Below, I go through how to use the programs in this repository to complete these two steps. 

## Part One: Selecting Targets
In order to orthorectify the videos on each camera and line them up in space, the x and y pixel coordinates of targets within each cameras frame must be known and related to real world coordinates. Included in this repository is a program to find the pixel locations of targets with an image or video stream (as long and the targets are not moving in relation to the camera). To use it, run the file "*gcp_selector.py*". When you do this, a gui will open up and look like the image below. This window can be resized as needed for your computer. 

![gcp_selector_gui_open](https://github.com/user-attachments/assets/bd62185f-203f-42fc-aa06-11f1561e43b3)

The first step in using this program is to load a video or image of the same camera settings as the videos you are interested in orthorectifying into the program. To do this, click on the **Load Image/Video** button in the top right of the window. Navigate your file explorer in the window that pops up and open the image/video file of your choosing. Ideally, this image/video should be of the flume with no water on top of it so that the position of the targets are not distorted by the refraction caused by water. Once the file is loaded, the the left portion of the window should populate with the image or the first frame of the video. It will look like the image below. You can zoom in on this image by scrolling and pan by clicking and dragging.

![gcp_selector_gui_image_opened](https://github.com/user-attachments/assets/b5669baf-92da-4993-8ff7-b7f6eadc2298)

Now you can add targets by clicking the **Add Target** button. When you click this button, an empty target object is added in the menu to the right of the image (below the button panel). Each target object has three information boxes: **Target:** name of the target, **X Pixels:** x coordinate (in pixels) of the target point, and **Y Pixels:** y coordiante (in pixels) of the target point. This information can be added manually by clicking on an information box and typing in a value. Coordinate information can also be added by selecting the location of the point on the image. 

To fill out the information for a target, first, select the target by clicking on the button on to the left of the **Target:** information box. When selected it should look like this: 

![selected_empty target](https://github.com/user-attachments/assets/c2052697-7dc0-4000-92aa-1493054f35e4)

Now zoom in on a target in the image and double click to place the marker at the proper point. A red point will appear on the image in the location where you clicked. If you miss or would like to change the position of the point, you can double click again to move the point. Finally, assign the target its name by filling in the **Target:** information box accordingly. Repeat this process until all of the targets have been selected. **MAKE SURE TO SELECT THE TARGET THAT YOU WANT TO EDIT BEFORE DOUBLE CLICKING ON THE IMAGE**. The currently selected target will have its button selected in the menu and will appear as red on the image, while all others will appear as blue. If the target is new, there will be no point on the image yet, so no points will be displayed in red. If you manually change the coordinates of a target, you must unselect and reselect that target for the change to appear on the image. 

![gcp_selector_gui_gcps_and_points](https://github.com/user-attachments/assets/144251c3-49c5-41d6-96ca-23443677723f)

Above, if the user were to double click in a new location on the image, the red point on the image would move to the new location and the x and y pixel values would change in target "c" in the menu.

Once all of the targets have been added and you are satisfied, save the points by clicking on the **Save Target Coordinates** button. Navigate the file explorer to the folder where you would like to save the target coordinates, add a logical file name (eg. 20240529_exp1_Cam1_targets.csv"), and save the file. This will save the coordinates as a csv file (see image below) which you can then go into and add the corresponding real world coordinates to. 

![example_output](https://github.com/user-attachments/assets/0107a286-421a-4ad5-8a3f-196a4c60f2a8)

When adding the real world coordinates to this file, use column names shown in the image below.
![image](https://github.com/user-attachments/assets/d1fd17fa-c1ca-4b65-a946-2d6794ee7f49)

You can also add points from a file to edit their locations without starting from scrath. To do this, you can upload a csv file with the above columns. Keep in mind that when it saves, it will only save the image coordinates and you will have to add in the real world coordinates again before using the file in the orthorecitfication code. Note that you may be able to use a single set of target files for all of your experiments. Unless the camera or the targets are physically moved or the camera settings are chagned, the relative position of the targets and the camera do not change, so the pixel locations of targets in the image from each camera are also not going to change.


## Part Two: Orthorectifying Videos
Once you have your target files (which relate image coordinates of targets to real world coordinates of targets), you can run the main program included in this repository. To run this program, we first need to change a few settings within the script (I did not make a GUI for this program, so there is a little bit of script editing that you will need to do). To start, open the "*orthomosaic_videos.py*" file. This file is 104 lines of code long, but as a user, the only lines you are interested in are lines 18-23. This is where are the imput variables are housed. 

![variables_to_edit](https://github.com/user-attachments/assets/919c25de-c4bc-440e-82f4-cf4728f13f4b)
