#import neccesary packages and modules
import pandas as pd
import cv2

#video file from camera 1 to open first frame
cam1_vid = "C:/Users/jwelsh/Image Annotation/20240529_Video_to_Frames/SAFL_Cam1/GX060169.MP4"

#gcps file from first camera
cam1_gcps = "C:/Users/jwelsh/Image Annotation/20240529_Video_to_Frames/Cam1_gcps.csv"


#load gcps into pandas dataframe
gcps_df = pd.read_csv(cam1_gcps)
print(gcps_df)

#load first valid frame of video 
# Open the video
cap = cv2.VideoCapture(cam1_vid)

if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

first_frame = None

# Loop until we find the first valid frame
while True:
    ret, frame = cap.read()
    if not ret:
        print("No valid frames found in the video.")
        break
    if frame is not None:
        first_frame = frame
        break

cap.release()

# Display the frame in a smaller window
if first_frame is not None:
    # Create a resizable window
    cv2.namedWindow("First Frame", cv2.WINDOW_NORMAL)
    
    # Resize the window (width, height)
    cv2.resizeWindow("First Frame", 640, 360)
    
    cv2.imshow("First Frame", first_frame)
    
    while True:
        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC key
            break
    cv2.destroyAllWindows()

