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
            
            # Initialize model architecture
            self.model = AutoModelForImageClassification.from_pretrained(
                "microsoft/resnet-50",
                num_labels=2,  # Binary classification: safe vs unsafe
                ignore_mismatched_sizes=True
            )
            self.model.load_state_dict(state_dict)
            self.model.to(self.device)
            self.model.eval()
            
            logger.info("Model loaded successfully")
            
        except Exception as e:
            logger.error(f"Error initializing ContentModerator: {str(e)}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            raise
    
    def analyze_frames(self, frames):
        """
        Analyze a list of video frames for inappropriate content.
        
        Args:
            frames (list): List of numpy arrays representing video frames
            
        Returns:
            list: List of dictionaries containing analysis results for each frame
        """
        try:
            # Create dummy labels (all zeros) for the dataset
            labels = [0] * len(frames)
            
            # Create dataset and dataloader
            dataset = VideoFrameDataset(frames, labels, None)  # feature_extractor is not needed anymore
            dataloader = DataLoader(dataset, batch_size=32, shuffle=False)
            
            results = []
            self.model.eval()
            
            with torch.no_grad():
                for batch in dataloader:
                    # Move batch to device
                    pixel_values = batch['pixel_values'].to(self.device)
                    
                    # Get model predictions
                    outputs = self.model(pixel_values)
                    predictions = torch.softmax(outputs.logits, dim=1)
                    
                    # Process predictions
                    for i in range(len(predictions)):
                        prob_unsafe = predictions[i][1].item()  # Probability of unsafe content
                        results.append({
                            'flagged': prob_unsafe > 0.5,
                            'confidence': prob_unsafe,
                            'reason': 'Unsafe content detected' if prob_unsafe > 0.5 else 'Content appears safe'
                        })
            
            return results
            
        except Exception as e:
            logger.error(f"Error analyzing frames: {str(e)}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            raise 
