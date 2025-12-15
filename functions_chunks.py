#functions.py

'''A file that holds all of the functions used in the process of orthomosaicing the video files.'''

#import neccesary packages and modules
from pathlib import Path
import csv
import tkinter as tk
from tkinter import filedialog
import tempfile
import sys
import moviepy.editor as mp
from scipy.signal import correlate
from scipy.io import wavfile
import cv2
import numpy as np
import asyncio
import concurrent.futures



class File_Functions():
    '''Class containing methods for user handling of files.'''

    def __init__(self):
        print("Initialized File_Funtions")

    def load_dn(self, purpose):
        """this function opens a tkinter GUI for selecting a 
        directory and returns the full path to the directory 
        once selected
        
        'purpose' -- provides expanatory text in the GUI
        that tells the user what directory to select"""

        root = tk.Tk()
        root.withdraw()
        directory_name = filedialog.askdirectory(title = purpose)

        return directory_name

    def load_fn(self, purpose):
        """this function opens a tkinter GUI for selecting a 
        file and returns the full path to the file 
        once selected
        
        'purpose' -- provides expanatory text in the GUI
        that tells the user what file to select"""

        root = tk.Tk()
        root.withdraw()
        filename = filedialog.askopenfilename(title = purpose)

        return filename

    def get_sorted_video_filenames(self, directory):
        # Convert the directory to a Path object
        directory_path = Path(directory)
        
        # Get the list of subdirectories (one for each camera)
        subdirectories = [subdir for subdir in directory_path.iterdir() if subdir.is_dir()]
        
        # Initialize a list to store the lists of filenames for each camera
        camera_files = []
        
        # Iterate through each camera subdirectory
        for subdir in subdirectories:
            # Get all .MP4 files in the subdirectory and store their full paths
            mp4_files = [file for file in subdir.iterdir() if file.suffix == ".MP4"]
            
            # Sort the filenames based on their name
            mp4_files.sort()
            
            # Add the sorted list of full paths to camera_files
            camera_files.append([str(file) for file in mp4_files])

        return camera_files
    
    def get_gcps_files(self, directory):
        # Convert directory to Path object
        directory_path = Path(directory)
        
        # Get all CSV files in the directory and sort them alphabetically
        gcps_files = sorted(directory_path.glob("*.csv"))
        
        # Convert Path objects to strings with consistent forward slashes
        gcps_files = [str(file_path) for file_path in gcps_files]
        
        return gcps_files

    def import_gcps(self, gcps_fn):
        """module for importing ground control points as lists"""

        gcps_rw_list = [] #make list for real world coordinates of GCPs
        gcps_image_list = [] #make list for image coordinates of GCPs

        #Read csv file into a list of real world and a list of image gcp coordinates
        with open(gcps_fn, 'r', newline='') as csvfile:
            # Create a CSV reader object
            csv_reader = csv.reader(csvfile)

            # Skip the header row
            next(csv_reader)

            # Iterate over each row in the CSV file
            for row in csv_reader:
                # Each row is a list where each element represents a column value
                gcps_image_list.append(row[1:3])
                gcps_rw_list.append(row[3:5])

                gcps = [gcps_rw_list, gcps_image_list]

        return gcps
    
class Audio_Functions():
    def __init__(self):
        self.loop = asyncio.get_event_loop()
        self.executor = concurrent.futures.ThreadPoolExecutor()
        print("Initialized Audio_Functions")

    async def extract_audio_async(self, video_path):
        """Asynchronously extract audio from an MP4 and save it as a temporary WAV file."""
        return await self.loop.run_in_executor(self.executor, self.extract_audio, video_path)

    def extract_audio(self, video_path):
        """This method extracts audio from an MP4 and saves it as a temporary WAV file."""
        print(video_path)
        clip = mp.VideoFileClip(video_path)
        audio = clip.audio
        # Save audio to a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as temp_audio_file:
            temp_audio_path = temp_audio_file.name
        audio.write_audiofile(temp_audio_path, codec='pcm_s16le')
        rate, audio_data = wavfile.read(temp_audio_path)
        clip.reader.close()
        return rate, audio_data

    async def extract_all_audios(self, video_paths):
        """Run extract_audio asynchronously for a list of video paths."""
        tasks = [self.extract_audio_async(video_path) for video_path in video_paths]
        return await asyncio.gather(*tasks)

    def find_time_offset(self, rate1, audio1, rate2, audio2):
        """This function compares two audio files and lines up the wave patterns to match in time.
        Returns the time offset."""

        # Ensure the sample rates are the same
        if rate1 != rate2:
            raise ValueError("Sample rates of the two audio tracks do not match")

        # Convert stereo to mono if necessary
        if len(audio1.shape) == 2:
            audio1 = audio1.mean(axis=1)
        if len(audio2.shape) == 2:
            audio2 = audio2.mean(axis=1)

        # Normalize audio data to avoid overflow
        audio1 = audio1 / np.max(np.abs(audio1))
        audio2 = audio2 / np.max(np.abs(audio2))

        # Compute cross-correlation
        correlation = correlate(audio1, audio2)
        lag = np.argmax(correlation) - len(audio2) + 1

        # Calculate the time offset in milliseconds
        time_offset = lag / rate1 * 1000

        return time_offset

    def find_all_offsets(self, rate_data, audio_data):
        """Find time offsets between the first audio and the others concurrently."""
        tasks = []
        for i in range(1, len(audio_data)):
            task = self.executor.submit(self.find_time_offset, rate_data[i], audio_data[i], rate_data[i-1], audio_data[i-1])
            tasks.append(task)

        time_offsets = [0]
        for i, task in enumerate(tasks):
            offset = task.result() + time_offsets[i]
            time_offsets.append(offset)
            print(f"Offset between video {i+1} and video {i+2} found")

        return time_offsets

class Video_Functions():
    def __init__(self):
        print("Initialized Video_Functions.")

    def find_homography(self, cam, gcps):
        """Method for finding homography matrix."""

        #adjust the ground control points so that they are within the frame of the camera, which starts at (0,0) for each camera
        for count, i in enumerate(gcps[0]):
            i[0] = float(i[0]) - 2438 * (cam-1)
            i[1] = (float(i[1])*-1) + 2000

        #convert the image and destination coordinates to numpy array with float32
        src_pts = np.array(gcps[1])
        src_pts = np.float32(src_pts[:, np.newaxis, :])

        dst_pts = np.array(gcps[0])
        dst_pts = np.float32(dst_pts[:, np.newaxis, :])

        #now we can find homography matrix
        h_matrix = cv2.findHomography(src_pts, dst_pts)

        return h_matrix[0]
    
    def calculate_reprojection_error(self, gcps, homography_matrix, x_range=None, cam=0):
        """
        Calculate the reprojection error given ground control points and a homography matrix.
        
        Args:
            gcps (tuple): A tuple containing two sets of points. 
                        gcps[0] - Real-world points (x, y)
                        gcps[1] - Image points (x, y)
            homography_matrix (numpy array): 3x3 Homography matrix for transformation.
            x_range (tuple, optional): A tuple specifying the range of x coordinates (min_x, max_x) to filter the real-world points.
        
        Returns:
            float: Mean reprojection error for the selected points.
        """

        # Convert to float32 numpy arrays
        img_pts = np.array(gcps[1], dtype=np.float32)
        rw_pts = np.array(gcps[0], dtype=np.float32)

        # Add a third dimension to the image points (homogeneous coordinates)
        img_pts = np.column_stack((img_pts, np.ones((img_pts.shape[0], 1))))

        # Apply homography to the source points
        projected_pts = homography_matrix @ img_pts.T

        # Normalize the projected points by dividing by the third coordinate
        projected_pts = projected_pts[:2] / projected_pts[2]

        # Transpose projected points for easier manipulation (back to Nx2)
        projected_pts = projected_pts.T
        
        # If an x_range is provided, filter points based on the real-world x coordinate
        if x_range is not None:
            min_x, max_x = x_range
            # Create a mask based on the x-coordinates of real-world points
            mask = (rw_pts[:, 0] >= min_x) & (rw_pts[:, 0] <= max_x)
            
            # Apply the mask to real-world points
            rw_pts = rw_pts[mask]
            
            # Apply the same mask to the corresponding projected points
            projected_pts = projected_pts[mask]

        # Calculate the Euclidean distance between actual and projected points
        errors = np.linalg.norm(rw_pts - projected_pts, axis=1)

        # Return the average error
        mean_error = np.mean(errors)
        print(f"Reprojection Error for cam {cam+1}: {mean_error}")
        
        return mean_error

    def frame_to_umat_frame(self, frame):
        uframe = cv2.UMat(frame)
        return uframe

    def orthomosaicing(self, captures_list, time_offsets, homo_mats, out_vid_dn, OUT_NAME, SPEED, START_TIME, LENGTH, COMPRESSION):
        # --- Video shapes ---
        final_shape = [2438, 4000]
        compressed_shape = (int(final_shape[0] / COMPRESSION), int(final_shape[1] / COMPRESSION))
        output_shape = (compressed_shape[0] * 4, compressed_shape[1])
        print("out Shape: ", output_shape)

        # --- Codec ---
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Windows-friendly

        # --- Initialize captures ---
        current_caps = [captures[0] for captures in captures_list]
        capture_indices = [0] * len(captures_list)
        frame_rates = [cap.get(cv2.CAP_PROP_FPS) for cap in current_caps]
        frame_counters = [0] * len(captures_list)

        # --- Prepare black frame ---
        ret, first_frame = current_caps[0].read()
        if not ret or first_frame is None:
            print("Error: Could not read the first frame.")
            sys.exit()
        first_frame_resized = cv2.resize(first_frame, compressed_shape)
        black_frame = np.zeros(first_frame_resized.shape, dtype=first_frame_resized.dtype)

        # --- Reset start positions ---
        start_time_ms = START_TIME * 1000
        for i, cap in enumerate(current_caps):
            cap.set(cv2.CAP_PROP_POS_MSEC, start_time_ms + time_offsets[i])

        # --- Chunked writer setup ---
        base_out = OUT_NAME.replace(".mp4", "")
        part = 1
        MAX_BYTES = 4 * 1024 * 1024 * 1024  # 4 GB

        def new_writer(part, frame_rate):
            filename = f"{base_out}_part{part:03d}.mp4"
            print(f"Starting new video file: {filename}")
            return cv2.VideoWriter(
                str(Path(out_vid_dn) / filename),
                fourcc,
                frame_rate * SPEED,
                output_shape
            ), filename

        out, current_filename = new_writer(part, frame_rates[0])

        # --- Frame processing function ---
        def process_frame(cap, homo_mat, black_frame, capture_index, capture_indices, frame_counters):
            ret, frame = cap.read()
            if frame is None or frame.size == 0:
                # Handle empty frames
                if cap.get(cv2.CAP_PROP_POS_MSEC) == 0.0 and frame_counters[capture_index] < 5 * 24:
                    frame_counters[capture_index] += 1
                    return black_frame, capture_index, frame_counters[capture_index]

                capture_indices[capture_index] += 1
                if capture_indices[capture_index] < len(captures_list[capture_index]):
                    next_cap = captures_list[capture_index][capture_indices[capture_index]]
                    current_caps[capture_index] = next_cap
                    frame_counters[capture_index] = 0
                    ret2, frame2 = next_cap.read()
                    if ret2 and frame2 is not None:
                        uframe = cv2.UMat(frame2)
                        corrected = cv2.warpPerspective(uframe, homo_mat, final_shape)
                        corrected = cv2.resize(corrected, compressed_shape)
                        return corrected.get(), capture_index, frame_counters[capture_index]
                return black_frame, capture_index, frame_counters[capture_index]

            uframe = cv2.UMat(frame)
            corrected = cv2.warpPerspective(uframe, homo_mat, final_shape)
            corrected = cv2.resize(corrected, compressed_shape)
            frame_counters[capture_index] += 1
            return corrected.get(), capture_index, frame_counters[capture_index]

        # --- Main processing loop ---
        count = 0
        while count <= LENGTH:
            args = [(current_caps[i], homo_mats[i], black_frame, i, capture_indices, frame_counters) for i in range(len(current_caps))]
            with concurrent.futures.ThreadPoolExecutor() as executor:
                results = list(executor.map(lambda p: process_frame(*p), args))

            corrected_frames = [f for f, _, _ in results]
            if corrected_frames:
                merged = cv2.hconcat(corrected_frames)
                out.write(merged)

                # --- Check for 4GB split ---
                file_path = Path(out_vid_dn) / current_filename
                if file_path.exists() and file_path.stat().st_size >= MAX_BYTES:
                    print(f"Reached 4GB limit. Closing {current_filename}...")
                    out.release()
                    part += 1
                    out, current_filename = new_writer(part, frame_rates[0])

            count += 1 / frame_rates[0]
            print(f"Processed {count:.2f} seconds.")

        # --- Release everything ---
        for cap in current_caps:
            if cap:
                cap.release()
        out.release()
        cv2.destroyAllWindows()

