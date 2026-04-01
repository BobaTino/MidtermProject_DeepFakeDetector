import cv2
import os

INPUT_DIRS = {
    "real": "sample/videos_real",
    "fake": "sample/videos_fake"
}

OUTPUT_DIR = "dataset"
FRAMES_PER_VIDEO = 10

def extract_frames(video_path, output_folder):
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    step = max(total_frames // FRAMES_PER_VIDEO, 1)
    
    count = 0
    saved = 0

    video_id = os.path.splitext(os.path.basename(video_path))[0]

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if count % step == 0 and saved < FRAMES_PER_VIDEO:
            frame = cv2.resize(frame, (224, 224))
            
            filename = os.path.join(output_folder, f"{video_id}_{saved}.jpg")
            cv2.imwrite(filename, frame)
            
            saved += 1

        count += 1

    cap.release()

def process_videos():
    for label, folder in INPUT_DIRS.items():
        output_folder = os.path.join(OUTPUT_DIR, label)
        os.makedirs(output_folder, exist_ok=True)

        for video_name in os.listdir(folder):
            video_path = os.path.join(folder, video_name)
            
            if video_path.endswith((".mp4", ".avi", ".mov")):
                print(f"Processing {video_name}...")
                extract_frames(video_path, output_folder)

process_videos()