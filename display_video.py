import cv2
from functions import File_Functions

ff = File_Functions()

# Open the video file or capture from a webcam (use 0 for webcam)
video_path = ff.load_fn("Choose video to display")
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

# Create a resizable window
cv2.namedWindow("RGB and Grayscale Video", cv2.WINDOW_NORMAL)

while cap.isOpened():
    ret, frame = cap.read()

    # If frame read was successful
    if ret:
        # Convert to grayscale and stack the frames vertically
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray_frame = cv2.cvtColor(gray_frame, cv2.COLOR_GRAY2BGR)  # Convert grayscale to BGR for stacking
        stacked_frame = cv2.vconcat([frame, gray_frame])

        # Get current window dimensions
        window_width, window_height = cv2.getWindowImageRect("RGB and Grayscale Video")[2:4]

        # Get dimensions of the stacked frame
        height, width = stacked_frame.shape[:2]

        # Calculate scale factor to maintain aspect ratio
        scale_factor = min(window_width / width, window_height / height)
        new_width = int(width * scale_factor)
        new_height = int(height * scale_factor)

        # Center the resized frame in the window if the window size is larger
        black_frame = cv2.resize(stacked_frame, (new_width, new_height), interpolation=cv2.INTER_AREA)
        display_frame = cv2.copyMakeBorder(
            black_frame,
            top=(window_height - new_height) // 2,
            bottom=(window_height - new_height) // 2,
            left=(window_width - new_width) // 2,
            right=(window_width - new_width) // 2,
            borderType=cv2.BORDER_CONSTANT,
            value=[0, 0, 0]
        )

        # Display the final output
        cv2.imshow("RGB and Grayscale Video", display_frame)

        # Break loop on 'q' key press
        if cv2.waitKey(30) & 0xFF == ord('q'):
            break
    else:
        break

# Release video capture object and close all windows
cap.release()
cv2.destroyAllWindows()