import torch
import numpy as np
import os
import json
from torchvision import transforms
from datetime import datetime
from model import load_model
from utils import extract_frames

# Load model
model = load_model()

# Transform
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

def predict_video(video_path):
    frames = extract_frames(video_path, max_frames=20)

    fake_scores = []

    for frame in frames:
        input_tensor = transform(frame).unsqueeze(0)

        with torch.no_grad():
            output = model(input_tensor)
            prob = torch.softmax(output, dim=1)

            fake_prob = prob[0][0].item()  # 0 = fake
            fake_scores.append(fake_prob)

    avg_score = np.mean(fake_scores)

    label = "FAKE" if avg_score > 0.5 else "REAL"
    confidence = avg_score * 100  # convert to %

    return label, confidence


# Batch processing
def process_folder(folder_path):
    results = []

    for file in os.listdir(folder_path):
        if file.endswith((".mp4", ".avi", ".mov")):
            video_path = os.path.join(folder_path, file)
            print(f"\nProcessing: {file}")

            label, confidence = predict_video(video_path)

            print(f"Prediction: {label}")
            print(f"Confidence: {confidence:.2f}%")

            results.append({
                "video": file,
                "prediction": label,
                "confidence": round(confidence, 2)
            })

    return results


# Save results
def save_results(results):
    output_dir = "results"
    os.makedirs(output_dir, exist_ok=True)

    # Optional: timestamp (so files don’t overwrite)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    json_path = os.path.join(output_dir, f"results_{timestamp}.json")
    txt_path = os.path.join(output_dir, f"results_{timestamp}.txt")

    # Save JSON
    with open(json_path, "w") as f:
        json.dump(results, f, indent=4)

    # Save TXT
    with open(txt_path, "w") as f:
        for r in results:
            f.write(f"{r['video']} - {r['prediction']} ({r['confidence']}%)\n")

    print(f"\nResults saved to:\n{json_path}\n{txt_path}")

if __name__ == "__main__":
    #Folder path
    folder = "sample\\videos_real"  

    results = process_folder(folder)
    save_results(results)