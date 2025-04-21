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
                
            # Load the model weights
            model_path = os.path.join(model_dir, 'model.safetensors')
            state_dict = load_file(model_path)
            
            # Initialize the model with the correct architecture
            self.model = AutoModelForImageClassification.from_pretrained(
                "models/best_model",
                state_dict=state_dict,
                local_files_only=True
            )
            
            self.model = self.model.to(self.device)
            self.model.eval()
            
            # Set up image preprocessing
            self.feature_extractor = AutoFeatureExtractor.from_pretrained(
                "models/best_model",
                local_files_only=True
            )
            
            logger.info("Model loaded successfully")
            
        except Exception as e:
            logger.error(f"Error initializing ContentModerator: {str(e)}")
            raise

    def preprocess_image(self, image):
        """Convert numpy array to PIL Image and apply transforms"""
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        return self.feature_extractor(image, return_tensors="pt").pixel_values.to(self.device)

    def analyze_frames(self, frames):
        """Analyze a list of frames for inappropriate content"""
        try:
            results = []
            batch_size = 32  # Process 32 frames at a time
            
            # Process frames in batches
            for i in range(0, len(frames), batch_size):
                batch = frames[i:i + batch_size]
                
                # Preprocess all frames in the batch
                inputs = self.feature_extractor(batch, return_tensors="pt").to(self.device)
                
                # Get model predictions for the entire batch
                with torch.no_grad():
                    outputs = self.model(**inputs)
                    logits = outputs.logits
                    probabilities = torch.softmax(logits, dim=1)
                    confidences, predicted = torch.max(probabilities, 1)
                    
                # Process results for each frame in the batch
                for j in range(len(batch)):
                    confidence = confidences[j].item()
                    pred = predicted[j].item()
                    is_unsafe = pred == 1  # Assuming class 1 is unsafe
                    
                    results.append({
                        'flagged': is_unsafe,
                        'confidence': confidence,
                        'reason': 'Inappropriate content detected' if is_unsafe else 'Content appears safe'
                    })
                    
            return results
            
        except Exception as e:
            logger.error(f"Error analyzing frames: {str(e)}")
            raise 
