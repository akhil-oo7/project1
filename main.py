import os
from video_processor import VideoProcessor
from content_moderator import ContentModerator
from dotenv import load_dotenv

def main():
    # Load environment variables
    load_dotenv()
    
    # Initialize components
    video_processor = VideoProcessor()
    content_moderator = ContentModerator()
    
    # Get video path from user
    video_path = input("Enter the path to the video file: ")
    
    if not os.path.exists(video_path):
        print("Error: Video file not found!")
        return
    
    try:
        # Process video and get frames
        frames = video_processor.extract_frames(video_path)
        
        # Analyze frames for content moderation
        results = content_moderator.analyze_frames(frames)
        
        # Display results
        print("\nContent Moderation Results:")
        print("--------------------------")
        print(f"Total Frames Analyzed: {len(results)}")
        print("\nSafety Analysis:")
        
        # Calculate overall video safety
        unsafe_frames = [r for r in results if r['flagged']]
        total_frames = len(results)
        unsafe_percentage = (len(unsafe_frames) / total_frames) * 100
        
        if unsafe_frames:
            print(f"Safe Content: {100.0 - unsafe_percentage:.2f}%")
            print(f"Unsafe Content: {unsafe_percentage:.2f}%")
            print("\nUnsafe Content Details:")
            for frame_idx, result in enumerate(results):
                if result['flagged']:
                    print(f"Frame {frame_idx+1}: {result['reason']} (Confidence: {result['confidence']:.2f})")
        else:
            print("Safe: 100%")
            print("Unsafe: 0%")
            print("No inappropriate content detected.")
            
    except Exception as e:
        print(f"Error during processing: {str(e)}")

if __name__ == "__main__":
    main() 