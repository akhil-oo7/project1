import cv2
import numpy as np
from tqdm import tqdm
import logging
import traceback

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
        """
        try:
            logger.info(f"Opening video file: {video_path}")
            frames = []
            cap = cv2.VideoCapture(video_path)
            
            if not cap.isOpened():
                error_msg = f"Could not open video file: {video_path}"
                logger.error(error_msg)
                raise ValueError(error_msg)
            
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            logger.info(f"Total frames in video: {total_frames}")
            
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
                            # Convert BGR to RGB
                            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                            # Resize frame to target size
                            frame_resized = cv2.resize(frame_rgb, self.target_size)
                            frames.append(frame_resized)
                        except Exception as e:
                            logger.error(f"Error processing frame {frame_count}: {str(e)}")
                            logger.error(f"Traceback: {traceback.format_exc()}")
                            continue
                    
                    frame_count += 1
                    pbar.update(1)
            
            cap.release()
            
            if not frames:
                error_msg = "No frames were extracted from the video"
                logger.error(error_msg)
                raise ValueError(error_msg)
            
            logger.info(f"Successfully extracted {len(frames)} frames")
            return frames
            
        except Exception as e:
            logger.error(f"Error in extract_frames: {str(e)}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            if 'cap' in locals():
                cap.release()
            raise 