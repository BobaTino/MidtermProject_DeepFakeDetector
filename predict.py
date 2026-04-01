import torch
import numpy as np
from torchvision import transforms
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

            fake_prob = prob[0][0].item() 
            fake_scores.append(fake_prob)

    avg_score = np.mean(fake_scores)

    if avg_score > 0.5:
        return "FAKE", avg_score
    else:
        return "REAL", avg_score


if __name__ == "__main__":
    video_path = "sample\\videos_fake\\vs9.mp4"

    label, score = predict_video(video_path)

    print(f"\nPrediction: {label}")
    print(f"Confidence: {score:.2f}")