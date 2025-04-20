import cv2
import numpy as np
from tqdm import tqdm
import logging
import traceback
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class VideoProcessor:
    def __init__(self, frame_interval=30, target_size=(224, 224)):
        """
        Initialize the VideoProcessor.
        
        Args:
            frame_interval (int): Number of frames to skip between extractions
            target_size (tuple): Target size for frame resizing (height, width)
        """
        self.frame_interval = frame_interval
        self.target_size = target_size
        logger.info(f"Initialized VideoProcessor with frame_interval={frame_interval}, target_size={target_size}")
    
    def extract_frames(self, video_path):
        """
        Extract frames from a video file.
        
        Args:
            video_path (str): Path to the video file
            
        Returns:
            list: List of extracted frames as numpy arrays
            
        Raises:
            ValueError: If video file cannot be opened or is empty
            Exception: For other processing errors
        """
        try:
            if not os.path.exists(video_path):
                error_msg = f"Video file not found: {video_path}"
                logger.error(error_msg)
                raise FileNotFoundError(error_msg)
            
            logger.info(f"Opening video file: {video_path}")
            frames = []
            cap = cv2.VideoCapture(video_path)
            
            if not cap.isOpened():
                error_msg = f"Could not open video file: {video_path}"
                logger.error(error_msg)
                raise ValueError(error_msg)
            
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            duration = total_frames / fps if fps > 0 else 0
            
            logger.info(f"Video details - Total frames: {total_frames}, FPS: {fps:.2f}, Duration: {duration:.2f}s")
            
            if total_frames == 0:
                error_msg = "Video file appears to be empty or corrupted"
                logger.error(error_msg)
                raise ValueError(error_msg)
            
            with tqdm(total=total_frames, desc="Extracting frames") as pbar:
                frame_count = 0
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    
                    if frame_count % self.frame_interval == 0:
                        try:
                            # Resize frame to target size
                            frame = cv2.resize(frame, self.target_size)
                            frames.append(frame)
                        except Exception as e:
                            logger.warning(f"Error processing frame {frame_count}: {str(e)}")
                            continue
                    
                    frame_count += 1
                    pbar.update(1)
            
            cap.release()
            
            if not frames:
                error_msg = "No frames were successfully extracted from the video"
                logger.error(error_msg)
                raise ValueError(error_msg)
            
            logger.info(f"Successfully extracted {len(frames)} frames from video")
            return frames
            
        except Exception as e:
            logger.error(f"Error in extract_frames: {str(e)}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            raise 