import torch
import numpy as np
import os
import json
import hashlib
import shutil
from torchvision import transforms
from datetime import datetime
from getpass import getpass

from model import load_model
from utils import extract_frames

# User Config
ALLOWED_EXTENSIONS = (".mp4", ".avi", ".mov")
MAX_FILE_SIZE_MB = 500
UPLOAD_FOLDER = "uploads"
RESULT_FOLDER = "results"
LOG_FOLDER = "logs"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)
os.makedirs(LOG_FOLDER, exist_ok=True)

# Users Profile
USERS = {
    "TheMatrix": "admin123",
    "Tino": "user123"
}

# Login Function
def login():
    username = input("Username: ")
    password = getpass("Password: ")

    if username in USERS and USERS[username] == password:
        log_event(f"LOGIN SUCCESS: {username}")
        print("Login successful.\n")
        return True
    else:
        log_event(f"LOGIN FAILED: {username}")
        print("Invalid login.")
        return False

# Logging Function
def log_event(msg):
    path = os.path.join(LOG_FOLDER, "audit_log.txt")
    with open(path, "a") as f:
        f.write(f"[{datetime.now()}] {msg}\n")

# File Validation
def valid_file(path):
    ext = os.path.splitext(path)[1].lower()

    if ext not in ALLOWED_EXTENSIONS:
        return False

    size = os.path.getsize(path) / (1024 * 1024)

    if size > MAX_FILE_SIZE_MB:
        return False

    return True

# Model Hash check
def hash_file(path):
    h = hashlib.sha256()

    with open(path, "rb") as f:
        while chunk := f.read(8192):
            h.update(chunk)

    return h.hexdigest()

print("Model SHA256:", hash_file("model.pth"))

model = load_model()

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# Prediction Function
def predict_video(video_path):
    frames = extract_frames(video_path, max_frames=20)

    fake_scores = []

    for frame in frames:
        input_tensor = transform(frame).unsqueeze(0)

        with torch.no_grad():
            output = model(input_tensor)
            prob = torch.softmax(output, dim=1)

            fake_prob = prob[0][0].item()
            fake_scores.append(fake_prob)

    avg_score = np.mean(fake_scores)

    label = "FAKE" if avg_score > 0.5 else "REAL"
    confidence = avg_score * 100

    return label, confidence

# Batch Processing
def process_folder(folder_path):
    results = []

    for file in os.listdir(folder_path):

        if file.endswith(ALLOWED_EXTENSIONS):

            video_path = os.path.join(folder_path, file)

            if not valid_file(video_path):
                log_event(f"BLOCKED FILE: {file}")
                continue

            print(f"\nProcessing: {file}")

            label, confidence = predict_video(video_path)

            # Low confidence warning
            if confidence < 60:
                final_label = f"{label} (Low Confidence)"
            else:
                final_label = label

            print(f"Prediction: {final_label}")
            print(f"Confidence: {confidence:.2f}%")

            log_event(f"PREDICTION: {file} => {final_label} ({confidence:.2f}%)")

            results.append({
                "video": file,
                "prediction": final_label,
                "confidence": round(confidence, 2)
            })

    return results

# Save Results
def save_results(results):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    json_path = os.path.join(RESULT_FOLDER, f"results_{timestamp}.json")
    txt_path = os.path.join(RESULT_FOLDER, f"results_{timestamp}.txt")

    with open(json_path, "w") as f:
        json.dump(results, f, indent=4)

    with open(txt_path, "w") as f:
        for r in results:
            f.write(f"{r['video']} - {r['prediction']} ({r['confidence']}%)\n")

    print(f"\nResults saved to:\n{json_path}\n{txt_path}")

if __name__ == "__main__":

    if login():

        folder = "sample\\videos_fake"

        results = process_folder(folder)
        save_results(results)