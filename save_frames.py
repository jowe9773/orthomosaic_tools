import cv2
import os

def save_frames(video_path, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    exp_name = output_dir.split("/")[-1]

    cap = cv2.VideoCapture(video_path)

    # ---- NEW: skip first 3000 frames ----
    frame_idx = 4080 

    save_remaining = 0
    total_saved = 0
    MAX_SAVED = 125
    jump = 48 #2 seconds between frames

    while total_saved < MAX_SAVED:
        # ---- NEW: jump to the start of the next jump-frame block ----
        next_trigger = frame_idx + (jump - (frame_idx % jump))
        cap.set(cv2.CAP_PROP_POS_FRAMES, next_trigger)
        frame_idx = int(cap.get(cv2.CAP_PROP_POS_FRAMES))

        save_remaining = 1  # start a burst

        # read & save 5 frames in a row
        while save_remaining > 0 and total_saved < MAX_SAVED:
            ret, frame = cap.read()
            if not ret:
                break

            out_path = os.path.join(
                output_dir, f"{exp_name}_frame_{total_saved:03d}_src_{frame_idx:06d}.png"
            )
            cv2.imwrite(out_path, frame)

            save_remaining -= 1
            total_saved += 1
            frame_idx += 24  # move forward by 24 during the burst
            print(frame_idx)

    cap.release()
    print(f"Saved {total_saved} frames.")

# make list of videos to extract frames from
exps = ["20240529_exp2", "20240606_exp1", "20240619_exp1", "20240621_exp1", "20240627_exp1", "20240628_exp1", "20240708_exp1", "20240714_exp1", "20240716_exp1","20240717_exp1", "20240718_exp1", "20240722_exp1", "20240724_exp1", "20240801_exp1", "20240805_exp1"]
exps = ["20240724_exp1"]
#exps = ["20240729_exp1"]

for i, exp in enumerate(exps):
    video_path = f"F:/Videos/{exp}_goprodata_full.mp4"
    output_dir = f"F:/Frames/{exp}"
    save_frames(video_path, output_dir)