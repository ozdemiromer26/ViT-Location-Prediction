import os
import torch
import torch.nn as nn
import pandas as pd
from PIL import Image
from torchvision import transforms, models
from torchvision.models import vit_b_16, ViT_B_16_Weights
from tqdm import tqdm

weights = ViT_B_16_Weights.IMAGENET1K_SWAG_E2E_V1

# Hyperparameter configuration class
class Config:
    BATCH_SIZE = 64
    NUM_WORKERS = 0
    IMG_SIZE = 384
    DROPOUT = 0.4
    NUM_NEURONS = 256  # Number of neurons in the hidden layer
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    MODEL_LOAD_PATH = "workspace/best_model.pth"

# Define the model
class LatitudeLongitudeModel(nn.Module):
    def __init__(self):
        super(LatitudeLongitudeModel, self).__init__()

        # Load a pretrained EfficientNet model
        self.base_model = vit_b_16(weights=weights)
        
        # Replace the classifier with a custom head
        self.base_model.heads = nn.Sequential(
            nn.Linear(self.base_model.heads[0].in_features, Config.NUM_NEURONS),
            nn.ReLU(),
            nn.Dropout(Config.DROPOUT),
            nn.Linear(Config.NUM_NEURONS, 2)  # 2 outputs: latitude and longitude
        )

    def forward(self, x):
        return self.base_model(x)
    
# Define transformations
transform = weights.transforms() # Predefined transforms that match the pretrained model

def load_model(model_path: str, device: torch.device, num_classes: int = 3):
    """
    Load the trained model from the given path.
    Adjust the architecture and final layer to match your training setup.
    """
    # Using ViT as in your original code. Adjust if needed. 
    model = LatitudeLongitudeModel()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    return model

def main():
    # Configuration
    latitude_min = 41.1001236366018
    latitude_range = 0.010612474509322567
    longitude_min = 29.015327288005498
    longitude_range = 0.021001475024995386
    test_csv = "workspace/data_384/data_384/test.csv"
    test_dir = "workspace/data_384/data_384/test"
    
    # Load the model
    model = load_model(Config.MODEL_LOAD_PATH, Config.DEVICE)

    # Read test.csv
    df = pd.read_csv(test_csv, sep=';')

    # Iterate through each row in test.csv
    for _, row in tqdm(df.iterrows(), total=len(df)):
        filename = row['filename']
        image_path = os.path.join(test_dir, f"{filename}")

        if not os.path.isfile(image_path):
            raise FileNotFoundError(f"Image file not found: {image_path}")

        # Load and preprocess the image
        image = Image.open(image_path).convert('RGB')
        image = transform(image).unsqueeze(0).to(Config.DEVICE)

        # Inference
        with torch.no_grad():
            outputs = model(image)
            outputs = outputs * torch.tensor([latitude_range, longitude_range], device=Config.DEVICE) + torch.tensor([latitude_min, longitude_min], device=Config.DEVICE)
            df.loc[df['filename'] == filename, ['latitude', 'longitude']] = outputs.cpu().numpy()

    # Save the updated CSV
    df.to_csv("workspace/sumbission.csv", index=False)
    print("Predictions saved to test.csv!")

if __name__ == "__main__":
    main()