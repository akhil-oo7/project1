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

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class VideoFrameDataset(Dataset):
    def __init__(self, frames, labels, feature_extractor):
        self.frames = frames
        self.labels = labels
        self.feature_extractor = feature_extractor
    
    def __len__(self):
        return len(self.frames)
    
    def __getitem__(self, idx):
        frame = self.frames[idx]
        label = self.labels[idx]
        
        # Convert numpy array to PIL Image
        image = Image.fromarray(frame)
        
        # Preprocess the image
        inputs = self.feature_extractor(image, return_tensors="pt")
        
        return {
            'pixel_values': inputs['pixel_values'].squeeze(),
            'label': torch.tensor(label, dtype=torch.long)
        }

class ContentModerator:
    def __init__(self, model_path="models/best_model"):
        """
        Initialize the ContentModerator with a trained model.
        
        Args:
            model_path (str): Path to the trained model directory
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")
        
        try:
            # Check if model directory exists
            if not os.path.exists(model_path):
                error_msg = f"Model directory not found at {model_path}"
                logger.error(error_msg)
                logger.error(f"Current working directory: {os.getcwd()}")
                logger.error(f"Directory contents: {os.listdir('.')}")
                raise FileNotFoundError(error_msg)
            
            # Check for model files
            model_files = os.listdir(model_path)
            logger.info(f"Model files found: {model_files}")
            
            # Load model weights
            model_weights_path = os.path.join(model_path, "model.safetensors")
            if not os.path.exists(model_weights_path):
                error_msg = f"Model weights file not found at {model_weights_path}"
                logger.error(error_msg)
                raise FileNotFoundError(error_msg)
            
            # Load model configuration
            config_path = os.path.join(model_path, "config.json")
            if not os.path.exists(config_path):
                error_msg = f"Model config file not found at {config_path}"
                logger.error(error_msg)
                raise FileNotFoundError(error_msg)
            
            logger.info(f"Loading model from {model_weights_path}")
            # Load model weights from safetensors
            state_dict = load_file(model_weights_path)
            
            # Initialize model architecture (you'll need to adjust this based on your model)
            self.model = AutoModelForImageClassification.from_pretrained(
                "microsoft/resnet-50",
                num_labels=2,  # Binary classification: violent vs non-violent
                ignore_mismatched_sizes=True
            )
            self.model.load_state_dict(state_dict)
            self.model.to(self.device)
            self.model.eval()
            
            logger.info("Model loaded successfully")
            
            # Define image transformations
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            
        except Exception as e:
            logger.error(f"Error initializing ContentModerator: {str(e)}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            raise
    
    def analyze_frames(self, frames):
        """
        Analyze frames for inappropriate content.
        
        Args:
            frames (list): List of video frames as numpy arrays
            
        Returns:
            list: List of analysis results for each frame
        """
        results = []
        
        try:
            # Convert frames to dataset
            dataset = VideoFrameDataset(frames, [0] * len(frames), self.transform)
            dataloader = DataLoader(dataset, batch_size=32)
            
            self.model.eval()
            with torch.no_grad():
                for batch in dataloader:
                    pixel_values = batch['pixel_values'].to(self.device)
                    outputs = self.model(pixel_values)
                    predictions = torch.softmax(outputs.logits, dim=1)
                    
                    for pred in predictions:
                        # Get probability of violence (class 1)
                        violence_prob = pred[1].item()
                        # Lower threshold for violence detection
                        flagged = violence_prob > 0.3  # Changed from 0.5 to 0.3
                        
                        results.append({
                            'flagged': flagged,
                            'reason': "Detected violence" if flagged else "No inappropriate content detected",
                            'confidence': violence_prob if flagged else 1 - violence_prob
                        })
            
            return results
        except Exception as e:
            logger.error(f"Error analyzing frames: {str(e)}")
            raise 
