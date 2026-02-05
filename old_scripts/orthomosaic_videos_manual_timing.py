#learning_async.py

"import packages and modules"
from pathlib import Path
import pandas as pd
import time
import cv2
from old_scripts.functions import File_Functions, Video_Functions, Audio_Functions


#define a orthomosaic video function that will process a single experiment
class orthomosaic_video:

    def __init__(self, gcps_filenames, main_dir, out_location, out_name, offsets_list, compression = 2.5, speed = 1, start_time = 0, length = 60, ):
        # instantiate classes
        self.ff = File_Functions()
        self.af = Audio_Functions()
        self.vf = Video_Functions()

        # define files/directories
        self.gcps_filenames = gcps_filenames
        self.main_dir = main_dir
        self.out_vid_dir = out_location

        # define important variables
        self.out_name = out_name
        self.offsets_list = offsets_list
        self.compression = compression
        self.speed = speed
        self.start_time = start_time
        self.length = length

        print("orthomosaic_video initialized")

    def _load_targets(self):
        self.targets = []
        for i, gcps in enumerate(self.gcps_filenames):
            gcps = self.ff.import_gcps(gcps)
            self.targets.append(gcps)
        

    def _set_offsets(self):
        # Set offsets
        off_1 = self.offsets_list[0]                  #if video from downstream cam is late, add time
        off_2 = self.offsets_list[1]                  #if video from downstream cam is early, subtract time
        off_3 = self.offsets_list[2]                  #value is in milliseconds

        #Generate time offsets using the first video from each camera
        self.time_offsets = [0, 0+off_1, 0+off_1+off_2, 0+off_1+off_2+off_3]
        print("Time offsets for video streams:")
        print(self.time_offsets) 

    
    def _run_homography(self):
        print(self.main_dir)
        #make list of videos in each camera (will use these videos to find homography)
        self.camera_files = self.ff.get_sorted_video_filenames(self.main_dir)
        print(self.camera_files)

        #make a list of the first video file for each camera
        first_vids = [camera[0] for camera in self.camera_files if camera]
        print(first_vids)

        #define the range of x values that will be used in calculating the error (from 0 to the width of each frame, which in the case of our flume is 2438mm)
        self.x_range = (0, 2348)

        #Generate homography matricies
        self.homo_mats = []
        self.reprojection_errors = []
        for i, vid in enumerate(first_vids):
            homography = self.vf.find_homography(i+1, self.targets[i])
            error = self.vf.calculate_reprojection_error(self.targets[i], homography, x_range = self.x_range, cam = i)

            self.homo_mats.append(homography)
            self.reprojection_errors.append(error)

    def _set_captures_list(self):
        #open captures for each video
        self.captures_lists = []
        for i, vid_list in enumerate(self.camera_files):
            captures = []
            for j, file in enumerate(vid_list):   
                cap = cv2.VideoCapture(file)
                captures.append(cap)
            self.captures_lists.append(captures)

        

    def run_orthomosaic(self):
        # load targets
        self._load_targets()

        # Start measuring time
        start_time = time.time()

        # load time_offsets
        self._set_offsets()

        self._run_homography()

        self._set_captures_list()

        # Start measuring time
        start_vid_time = time.time()

        print(self.captures_lists)

        self.vf.orthomosaicing_test(self.captures_lists, self.time_offsets, self.homo_mats, self.out_vid_dir, self.out_name, self.speed, self.start_time, self.length, self.compression)

        # End measuring time
        end_time = time.time()

        # Calculate and print the elapsed time
        elapsed_time = end_time - start_time
        video_time = end_time - start_vid_time

        print(f"Total time taken: {elapsed_time:.2f} seconds")
        print(f"Time taken to process frames: {video_time:.2f} seconds")
        print("Reproection errors: ", self.reprojection_errors)
        print("Time offsets: ", self.time_offsets)

if __name__ == "__main__":
    ff = File_Functions()

    #get all video files
    main_directory = "G:/video_data"

    # get gcps files
    all_gcps_directory = "C:/Users/jwelsh/Image Annotation/annotated_gcps"

    #choose an output location for the final video
    out_vid_dir = "C:/Users/jwelsh/Image Annotation"
    #out_vid_dir = ff.load_dn("Select output location for the final video")

    #get offsets from offsets_csv
    offsets_csv_fn = "C:/Users/jwelsh/Image Annotation/time_offsets.csv"

    #choose an experiment (by name)
    exp_name = "20240731_exp1"
    exp_date = exp_name.split("_")[0]

    #find the video directory containing the experiment name
    root = Path(main_directory)
    matches = [p for p in root.rglob("*") if p.is_dir() and exp_name in p.name]
    video_dir = matches[0]
    
    #find the gcps directory containing the experiment name
    root = Path(all_gcps_directory)
    matches = [p for p in root.rglob("*") if p.is_dir() and exp_date in p.name]
    gcps_directory = matches[0]

    #make a list of GCPS file names for the experiment name
    gcps_filenames = ff.get_gcps_files(gcps_directory)

    #find the time offsets from the csv of times
    offsets_df = pd.read_csv(offsets_csv_fn)
    row = offsets_df.loc[offsets_df["exp_name"] == exp_name, ["Offset 1", "Offset 2", "Offset 3"]]
    offsets_list = row.iloc[0].tolist()
    print(offsets_list)

    offsets_list = [7504+100-50-25, 5494+100-25+25-10, 6040-250+50] #if its late add time

    out_name = exp_name + "_test.mp4"

    #initialize orthomosaic class
    ortho = orthomosaic_video(gcps_filenames, video_dir, out_vid_dir, out_name, offsets_list, compression= 2.5, speed = 1, start_time = 700 , length = 10)

    ortho.run_orthomosaic()

    print(offsets_list)

    