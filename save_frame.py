import cv2

video_path = ""

expeiment_name = video_path.split(".")[0].split("/")[-1]

print(expeiment_name)

cap = cv2.VideoCapture(video_path)

# Go to specific frame (e.g., frame number 100)
frame_number = 100
cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)

# Read the frame
ret, frame = cap.read()

if ret:
    # Save the frame as an image
    out_path = "C:/Users/jwelsh/Image Annotation/frames_for_GCP_selection"
    output_path = out_path + "/" + "20240716_3" + ".jpg"
    cv2.imwrite(output_path, frame)
    print(f"Frame {frame_number} saved to {output_path}")
else:
    print(f"Failed to read frame {frame_number}")

cap.release()