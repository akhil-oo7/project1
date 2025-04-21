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
    def __init__(self, frame_interval=10, target_size=(224, 224), min_frames=10, max_frames=30):
        """
        Initialize the VideoProcessor.
        
        Args:
            frame_interval (int): Number of frames to skip between extractions
            target_size (tuple): Target size for frame resizing (height, width)
            min_frames (int): Minimum number of frames to extract
            max_frames (int): Maximum number of frames to extract
        """
        self.frame_interval = frame_interval
        self.target_size = target_size
        self.min_frames = min_frames
        self.max_frames = max_frames
        logger.info(f"Initialized VideoProcessor with frame_interval={frame_interval}, target_size={target_size}, min_frames={min_frames}, max_frames={max_frames}")
    
    def extract_frames(self, video_path, num_frames=128):
        """Extract frames from video file"""
        try:
            frames = []
            cap = cv2.VideoCapture(video_path)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            
            # Calculate frame interval to get approximately num_frames
            if total_frames > num_frames:
                interval = total_frames // num_frames
            else:
                interval = 1
            
            logger.info(f"Total frames in video: {total_frames}")
            logger.info(f"Extracting frames with interval: {interval}")
            
            frame_count = 0
            extracted_count = 0
            
            with tqdm(total=min(total_frames, num_frames), desc="Extracting frames") as pbar:
                while cap.isOpened() and extracted_count < num_frames:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    
                    if frame_count % interval == 0:
                        frames.append(frame)
                        extracted_count += 1
                        pbar.update(1)
                    
                    frame_count += 1
                
            cap.release()
            logger.info(f"Successfully extracted {len(frames)} frames from video")
            return frames
            
        except Exception as e:
            logger.error(f"Error extracting frames: {str(e)}")
            raise

    def extract_frames_old(self, video_path):
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
            
            # Calculate optimal frame interval based on video length
            if total_frames > self.max_frames * self.frame_interval:
                self.frame_interval = total_frames // self.max_frames
            elif total_frames < self.min_frames:
                self.frame_interval = 1
            
            with tqdm(total=total_frames, desc="Extracting frames") as pbar:
                frame_count = 0
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    
                    if frame_count % self.frame_interval == 0:
                        try:
                            # Convert BGR to RGB
                            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
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
            
            # Ensure we have at least min_frames
            if len(frames) < self.min_frames:
                # If we have too few frames, reduce the interval and try again
                self.frame_interval = max(1, total_frames // self.min_frames)
                return self.extract_frames(video_path)
            
            logger.info(f"Successfully extracted {len(frames)} frames from video")
            return frames
            
        except Exception as e:
            logger.error(f"Error in extract_frames: {str(e)}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            raise 