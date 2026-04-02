import cv2
import os

INPUT_DIRS = {
    "real": "sample/videos_real",
    "fake": "sample/videos_fake"
}

OUTPUT_DIR = "dataset"
FRAMES_PER_VIDEO = 20

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

def extract_frames(video_path, output_folder, max_frames=10):
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    step = max(total_frames // max_frames, 1)

    count = 0
    saved = 0

    video_id = os.path.splitext(os.path.basename(video_path))[0]

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if count % step == 0 and saved < max_frames:

            # Convert to grayscale
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Detect faces
            faces = face_cascade.detectMultiScale(gray, 1.3, 5)

            # Loop through detected faces
            for (x, y, w, h) in faces:
                face = frame[y:y+h, x:x+w]
                face = cv2.resize(face, (224, 224))

                filename = os.path.join(output_folder, f"{video_id}_{saved}.jpg")
                cv2.imwrite(filename, face)

                saved += 1

                # Stop if enough frames
                if saved >= max_frames:
                    break

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