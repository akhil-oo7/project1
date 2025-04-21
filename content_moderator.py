from transformers import pipeline, AutoModelForImageClassification, AutoFeatureExtractor
import torch
from PIL import Image
import numpy as np
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from tqdm import tqdm
import os
import logging
import traceback
from safetensors.torch import load_file
from torchvision import transforms
import torch.nn as nn
import torchvision.models as models

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class VideoFrameDataset(Dataset):
    def __init__(self, frames, labels, feature_extractor):
        self.frames = frames
        self.labels = labels
        self.feature_extractor = feature_extractor
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    def __len__(self):
        return len(self.frames)
    
    def __getitem__(self, idx):
        frame = self.frames[idx]
        label = self.labels[idx]
        
        # Convert numpy array to PIL Image
        image = Image.fromarray(frame)
        
        # Apply transforms
        image_tensor = self.transform(image)
        
        return {
            'pixel_values': image_tensor,
            'label': torch.tensor(label, dtype=torch.long)
        }

class ContentModerator:
    def __init__(self):
        try:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            logger.info(f"Using device: {self.device}")
            
            # Load model configuration
            model_dir = "models/best_model"
            if not os.path.exists(model_dir):
                raise FileNotFoundError(f"Model directory not found at {model_dir}")
                
            model_files = os.listdir(model_dir)
            logger.info(f"Model files found: {model_files}")
            
            if 'model.safetensors' not in model_files:
                raise FileNotFoundError("Model file not found at models/best_model/model.safetensors")
                
            # Initialize ResNet model
            self.model = models.resnet50(pretrained=False)
            
            # Modify the final layer for our binary classification task
            num_features = self.model.fc.in_features
            self.model.fc = nn.Sequential(
                nn.Linear(num_features, 512),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(512, 2)  # 2 classes: safe and unsafe
            )
            
            # Load the model weights
            model_path = os.path.join(model_dir, 'model.safetensors')
            state_dict = load_file(model_path)
            
            # Load state dict, ignoring size mismatches
            self.model.load_state_dict(state_dict, strict=False)
            
            self.model = self.model.to(self.device)
            self.model.eval()
            
            # Set up image preprocessing
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                  std=[0.229, 0.224, 0.225])
            ])
            
            logger.info("Model loaded successfully")
            
        except Exception as e:
            logger.error(f"Error initializing ContentModerator: {str(e)}")
            raise

    def preprocess_image(self, image):
        """Convert numpy array to PIL Image and apply transforms"""
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        return self.transform(image).unsqueeze(0).to(self.device)

    def analyze_frames(self, frames):
        """Analyze a list of frames for inappropriate content"""
        try:
            results = []
            for frame in frames:
                # Preprocess the frame
                input_tensor = self.preprocess_image(frame)
                
                # Get model prediction
                with torch.no_grad():
                    output = self.model(input_tensor)
                    probabilities = torch.softmax(output, dim=1)
                    confidence, predicted = torch.max(probabilities, 1)
                    
                # Convert to Python types
                confidence = confidence.item()
                predicted = predicted.item()
                
                # Determine if content is unsafe (class 1)
                is_unsafe = predicted == 1
                
                results.append({
                    'flagged': is_unsafe,
                    'confidence': confidence,
                    'reason': 'Inappropriate content detected' if is_unsafe else 'Content appears safe'
                })
                
            return results
            
        except Exception as e:
            logger.error(f"Error analyzing frames: {str(e)}")
            raise 
