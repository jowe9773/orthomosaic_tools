import numpy as np
import cv2

class Orthomosaicing():
    def __init__(self):
        print("Initialized Video_Functions.")

    # Helper functions!
    def _find_homography(self, cam, gcps):
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
    
    def _create_black_frame(self, reference_frame):
        black_frame = np.zeros(reference_frame.shape, dtype=reference_frame.dtype)
        return black_frame
    
    def _frame_to_umat_frame(self, frame): #get frame into a format that can be sent to a GPU
        uframe = cv2.UMat(frame)
        return uframe

    def _orthocrop_frame(self, cap, homo_mat, final_shape, compressed_shape):
        ret, frame = cap.read() #read the frame

        uframe = cv2.UMat(frame)  # Convert frame to UMat for processing

        corrected_frame = cv2.warpPerspective(uframe, homo_mat, final_shape)  # Apply homography
        corrected_frame = cv2.resize(corrected_frame, compressed_shape)  # Resize frame

        return corrected_frame

    
    def orthomosaicing(self, captures_list, time_offsets, homo_mats, out_vid_dn, OUT_NAME, SPEED, START_TIME, LENGTH, COMPRESSION):
        # Describe shape
        final_shape = [2438, 4000]
        compressed_shape = (int(final_shape[0] / COMPRESSION), int(final_shape[1] / COMPRESSION))
        output_shape = (compressed_shape[0] * 4, compressed_shape[1])

        print("out Shape: ", output_shape)

        # Set up video writer
        fourcc = cv2.VideoWriter_fourcc(*'avc1')
        out = cv2.VideoWriter(out_vid_dn + "/" + OUT_NAME, fourcc, captures_list[0][0].get(cv2.CAP_PROP_FPS) * SPEED, output_shape)

        # Initialize variables for tracking current capture and file
        current_caps = [captures[0] for captures in captures_list]
        capture_indices = [0] * len(captures_list)  # Track which file in each list is being used
        frame_rates = [cap.get(cv2.CAP_PROP_FPS) for cap in current_caps]

        print("Current captures:", current_caps)
        
        # Initialize counters to track frames processed per capture
        frame_counters = [0] * len(captures_list)

        # Get first valid frame to determine dimensions and data type for black frame
        ret, first_frame = current_caps[0].read()
        if not ret or first_frame is None:
            print("Error: Could not read the first frame.")
            sys.exit()
        first_frame_resized = cv2.resize(first_frame, compressed_shape)
        black_frame = self.create_black_frame(first_frame_resized)  # Black frame now matches other frames

        # Reset start position after reading the first frame
        start_time = START_TIME * 1000
        for i, cap in enumerate(current_caps):
            cap.set(cv2.CAP_PROP_POS_MSEC, start_time + time_offsets[i])

        # Initialize processed frames count
        frames_processed = [0] * len(captures_list)

        # Process frames in parallel
        count = 0
        seconds = 0

        print(LENGTH)
        pbar = tqdm(total = LENGTH*frame_rates[0])

        while seconds <= LENGTH:
            with concurrent.futures.ThreadPoolExecutor() as executor:
                # Create a list of arguments for each camera's processing
                args = [(current_caps[i], homo_mats[i], black_frame, i, capture_indices, frame_counters) for i in range(len(current_caps))]

                # Process frames concurrently
                results = list(executor.map(lambda p: self.process_frame(*p), args))

            # Collect corrected frames and update current captures
            corrected_frames = []
            for corrected_frame, index, frame_counter in results:
                corrected_frames.append(corrected_frame)

            # If there are valid frames, merge and write them to the output video
            if corrected_frames:
                merged = cv2.hconcat(corrected_frames)
                out.write(merged)

            count += 1 
            seconds = count / frame_rates[0]

            pbar.update(1)

        pbar.close()

        # Release all captures and writer objects at the end
        for cap in current_caps:
            if cap:
                cap.release()
        out.release()
        cv2.destroyAllWindows()