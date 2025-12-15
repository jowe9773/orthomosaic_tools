#import neccesary packages and modules
"import packages and modules"
import os
from pathlib import Path
import time
import asyncio
from pprint import pprint
import cv2
from functions_chunks import File_Functions, Video_Functions, Audio_Functions

#instantiate classes as neccesary
ff = File_Functions()

#get gcps files containing the starting positions 
gcps_directory = ff.load_dn("select directory containing gcps files.")
gcps_filenames = ff.get_gcps_files(gcps_directory)
print(gcps_filenames)

targets = []
for i, gcps in enumerate(gcps_filenames):
    gcps = ff.import_gcps(gcps)
    targets.append(gcps)


print(targets)