# batch_orthomosaic.py

#import neccesary packages and modules
from pathlib import Path
import pandas as pd
from functions import File_Functions, Video_Functions, Audio_Functions
from orthomosaic_videos_manual_timing import orthomosaic_video

ff = File_Functions()

#get all video files
main_directory = "G:/video_data"

# get gcps files
all_gcps_directory = "C:/Users/jwelsh/Image Annotation/annotated_gcps"

#choose an output location for the final video
out_vid_dir = "F:/Videos"

#get offsets from offsets_csv
offsets_csv_fn = "F:/Videos/time_offsets_v1.csv"

offsets_df = pd.read_csv(offsets_csv_fn)
exps = offsets_df["exp_name"].tolist()


for i, exp in enumerate(exps):
    exp_name = exp + "_goprodata"
    print("Exp_name", exp_name)
    exp_date = exp.split("_")[0]
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
    row = offsets_df.loc[offsets_df["exp_name"] == exp, ["Start Time", "Length", "Offset 1", "Offset 2", "Offset 3"]]
    times_list = row.iloc[0].tolist()
    print(times_list)

    start_time = times_list[0]
    length = times_list[1]
    offsets_list = [times_list[2], times_list[3], times_list[4]]

    out_name = exp_name + "_full.mp4"

    #set video params
    compression = 2.5
    speed = 1

    #set up orthomosaic_video 
    ortho = orthomosaic_video(gcps_filenames= gcps_filenames,
                            main_dir= video_dir, 
                            out_location= out_vid_dir, 
                            out_name= out_name, 
                            offsets_list= offsets_list, 
                            compression = compression, 
                            speed= speed, 
                            start_time= start_time, 
                            length = length
                            )

    ortho.run_orthomosaic()
