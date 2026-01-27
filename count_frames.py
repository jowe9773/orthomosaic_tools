import cv2
cap = cv2.VideoCapture("G:/video_data/20240603_exp1_goprodata/Cam1/GX020171.MP4")

count = 0 

while True:
    ret, frame = cap.read()
    if count >120 and not ret:
        break

    count+= 1
    print(count)

cap.release()
print(count)