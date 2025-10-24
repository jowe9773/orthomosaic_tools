import os
import cv2
import numpy as np
import math
from collections import deque

class VideoProcessor:
    """Class for working with video data from the flume"""

    def __init__(self, video_filename):
        """ VideoProcessor(video_filename):
            Returns an instance of the VideoProcessor class for the filename that was entered.
        """
        self.window_name = video_filename

        # 1. Check that the path exists and is a file
        if not os.path.isfile(video_filename):
            print("File doesn't exist. Check spelling and path!")
            return

        # 2. Set the read limit higher:
        os.environ["OPENCV_FFMPEG_READ_ATTEMPTS"] = "10000"

        # 3. Try to open with OpenCV
        self.cap = cv2.VideoCapture(video_filename)
        if not self.cap.isOpened():
            print("The file is not a video. File should be a '.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv', '.m4v'")
            return 
        
        # If the capture is succesfully created, then print confirmation
        print(f"{video_filename} is valid")

    def frames(self):
        """Generator that yields frames from the video one by one."""
        frame_count = 0
        paused = False

        while True:
            if not paused:
                ret, frame = self.cap.read()
                if not ret:
                    print(f"Skipping unreadable frame at index {frame_count}")
                    frame_count += 1
                    continue

            yield frame, frame_count

            # Check if user requested exit or pause
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('p'):
                paused = not paused
            if cv2.getWindowProperty(self.window_name, cv2.WND_PROP_VISIBLE) < 1:
                print("Window closed by user")
                break

            frame_count += 1
    
    def layout_frames(self, frames):
        """
        Arrange a list of image frames into a roughly square grid and return a single concatenated frame.

        The function automatically calculates the number of rows and columns needed to 
        make a grid as square as possible. If the number of frames is less than the 
        required grid size, black frames are added to fill the remaining spaces.

        Parameters
        ----------
        frames : list of ndarray
            A list of frames (NumPy arrays, e.g., from OpenCV) to arrange in the grid.
            All frames must have the same shape (height, width, channels).

        Returns
        -------
        ndarray
            A single frame obtained by horizontally concatenating frames row-wise and 
            vertically concatenating the rows.

        Notes
        -----
        - Uses `cv2.hconcat` and `cv2.vconcat` for concatenation.
        - Adds black frames as placeholders if the number of frames does not fill the grid completely.
        - Prints the computed number of rows, columns, and frame indices during processing (can be removed in production).

        Example
        -------
        concated_frame = layout_frames([frame1, frame2, frame3, frame4])
        """

        #count number of frames (k)
        k = len(frames)

        #figure out the best number of columns and rows to get a roughly square grid
        cols = math.ceil(math.sqrt(k))   # roughly square grid
        rows = math.ceil(k / cols)


        #turn list of frames into a matrix of rows and columns. Fill empty space at end with black frames
        #make a black frame to use as buffer if k != cols*rows
        if k != cols*rows:
            # Create a black frame of the same size
            black_frame = np.zeros_like(frames[0])

            for i in range(cols*rows - k):
                frames.append(black_frame)


        concat_rows = []
        for row in range(rows):
            frames_to_hconcat = []
            for col in range(cols):
                frame_number = row*(cols) + col
                frames_to_hconcat.append(frames[frame_number])

            concat_row = cv2.hconcat(frames_to_hconcat)
            concat_rows.append(concat_row)

        concated = cv2.vconcat(concat_rows)

        return concated
    
    def view(self, window_name, frame):
        """
        Display a frame in a resizable OpenCV window while maintaining its aspect ratio.

        The function scales the input frame to fit within the current window size,
        preserving the original aspect ratio. The scaled frame is centered on a black
        canvas matching the window size.

        Parameters
        ----------
        window_name : str
            Name of the OpenCV window where the frame will be displayed.
        frame : ndarray
            The image frame to display. Can be grayscale (2D array) or color (3D array).

        Returns
        -------
        None
            The function displays the frame in the specified window and does not return a value.

        Notes
        -----
        - Uses `cv2.getWindowImageRect` to obtain the current window size. If unavailable,
        the frame size is used.
        - Maintains aspect ratio by scaling the frame to fit within the window.
        - Centers the resized frame on a black canvas of the window's dimensions.
        - Supports both grayscale and color frames.
        - Calls `cv2.imshow` to update the window display.

        Example
        -------
        view_layout('My Window', frame)
        """
        self.window_name = window_name

        # Make window resizable
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

        frame_h, frame_w = frame.shape[:2]

        # Get current window size
        try:
            x, y, win_w, win_h = cv2.getWindowImageRect(window_name)
        except AttributeError:
            win_w, win_h = frame_w, frame_h

        # Compute scaling factor to maintain aspect ratio
        scale = min(win_w / frame_w, win_h / frame_h)
        new_w = int(frame_w * scale)
        new_h = int(frame_h * scale)

        # Resize frame
        resized_frame = cv2.resize(frame, (new_w, new_h))

        # Create a black canvas of window size
        if len(frame.shape) == 2:
            canvas = np.zeros((win_h, win_w), dtype=frame.dtype)
        else:
            canvas = np.zeros((win_h, win_w, frame.shape[2]), dtype=frame.dtype)

        # Compute top-left corner for centering
        y_offset = (win_h - new_h) // 2
        x_offset = (win_w - new_w) // 2

        # Place the resized frame onto the canvas
        canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized_frame

        cv2.imshow(window_name, canvas)

    def grayscale(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray_BGR = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        return  gray, gray_BGR

    def extract_BGR(self, frame):
        b = frame[:, :, 0]
        g = frame[:, :, 1]
        r = frame[:, :, 2]

        b_BGR = cv2.cvtColor(b, cv2.COLOR_GRAY2BGR)
        g_BGR = cv2.cvtColor(g, cv2.COLOR_GRAY2BGR)
        r_BGR = cv2.cvtColor(r, cv2.COLOR_GRAY2BGR)

        return b, g, r, b_BGR, g_BGR, r_BGR

    def normalized_BGR(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        b = frame[:, :, 0]
        g = frame[:, :, 1]
        r = frame[:, :, 2]

        norm_b = b/gray
        norm_b_uint8 = cv2.convertScaleAbs(norm_b*255)
        norm_g = g/gray
        norm_g_uint8 = cv2.convertScaleAbs(norm_g*255)
        norm_r = r/gray
        norm_r_uint8 = cv2.convertScaleAbs(norm_r*255)

        norm_b_BGR = cv2.cvtColor(norm_b_uint8, cv2.COLOR_GRAY2BGR)
        norm_g_BGR = cv2.cvtColor(norm_g_uint8, cv2.COLOR_GRAY2BGR)
        norm_r_BGR = cv2.cvtColor(norm_r_uint8, cv2.COLOR_GRAY2BGR)
              
        return norm_b, norm_g, norm_r, norm_b_BGR, norm_g_BGR, norm_r_BGR
    
    def rolling_average_subtract(self, frame, buffer, window_size=50):
        """
        Subtracts the average grayscale of surrounding frames from the current frame.
        
        Parameters
        ----------
        frame : ndarray
            Current color frame (BGR)
        buffer : deque
            Deque storing previous grayscale frames for computing local mean
        window_size : int
            Number of frames to consider in the rolling average
        
        Returns
        -------
        ndarray
            Grayscale frame with local mean subtracted
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Add current frame to buffer
        buffer.append(gray.astype(np.float32))

        # Ensure buffer doesn't exceed window size
        if len(buffer) > window_size:
            buffer.popleft()

        # Compute mean over buffered frames
        avg_frame = np.mean(buffer, axis=0)

        # Subtract mean from current frame and scale to 0-255
        subtracted = gray.astype(np.float32) - avg_frame
        subtracted = cv2.normalize(subtracted, None, 0, 255, cv2.NORM_MINMAX)
        subtracted = subtracted.astype(np.uint8)

        # Convert to BGR for layout display
        subtracted_BGR = cv2.cvtColor(subtracted, cv2.COLOR_GRAY2BGR)

        return subtracted, subtracted_BGR
    

if __name__ == "__main__":
    video = VideoProcessor("GX020176.MP4")

    rolling_buffer = deque()
    n = 10

    for frame, index in video.frames():
        if index % n != 0:
            continue  # skip frames that are not every nth

        gray, gray_BGR = video.grayscale(frame)

        b, g, r, b_BGR, g_BGR, r_BGR = video.extract_BGR(frame)

        nb, ng, nr, nb_BGR, ng_BGR, nr_BGR = video.normalized_BGR(frame)

        subtracted, subtracted_BGR = video.rolling_average_subtract(nb_BGR, rolling_buffer, 500)



        combined = video.layout_frames([gray_BGR, b_BGR, nb_BGR, subtracted_BGR])
        

        video.view("greyscale and RGB", combined)


    video.cap.release()
    cv2.destroyAllWindows()