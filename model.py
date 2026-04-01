import torch
import torch.nn as nn
from torchvision import models

def load_model():
    model = models.resnet18(pretrained=False)
    model.fc = nn.Linear(model.fc.in_features, 2)

    model.load_state_dict(torch.load("model.pth", map_location=torch.device('cpu')))
    model.eval()
    
    return model