from flask import Flask, render_template, request, jsonify
import os
import logging
import base64
from werkzeug.utils import secure_filename
from video_processor import VideoProcessor
from content_moderator import ContentModerator
import cv2
import traceback

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = '/tmp'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Initialize components
video_processor = VideoProcessor()
try:
    content_moderator = ContentModerator()
    logger.info("ContentModerator initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize ContentModerator: {str(e)}")
    content_moderator = None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze_video():
    if 'video' not in request.files:
        logger.error("No video file in request")
        return jsonify({'error': 'No video file provided'}), 400
    
    file = request.files['video']
    if file.filename == '':
        logger.error("Empty filename")
        return jsonify({'error': 'No file selected'}), 400
    
    try:
        # Save the uploaded file
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        logger.info(f"File saved to {filepath}")
        
        # Process video
        frames = video_processor.extract_frames(filepath)
        logger.info(f"Extracted {len(frames)} frames")
        
        # Analyze frames
        results = content_moderator.analyze_frames(frames)
        logger.info(f"Analyzed {len(results)} frames")
        
        # Prepare response
        flagged_frames = []
        for i, result in enumerate(results):
            if result['flagged']:
                # Convert frame to base64 for display
                _, buffer = cv2.imencode('.jpg', frames[i])
                frame_base64 = base64.b64encode(buffer).decode('utf-8')
                
                flagged_frames.append({
                    'frame_index': i,
                    'reason': result['reason'],
                    'confidence': result['confidence'],
                    'image': frame_base64
                })
        
        # Calculate unsafe percentage
        total_frames = len(results)
        unsafe_frames = len(flagged_frames)
        unsafe_percentage = (unsafe_frames / total_frames) * 100 if total_frames > 0 else 0
        
        response = {
            'status': 'UNSAFE' if unsafe_frames > 0 else 'SAFE',
            'total_frames': total_frames,
            'unsafe_frames': unsafe_frames,
            'unsafe_percentage': unsafe_percentage,
            'flagged_frames': flagged_frames
        }
        
        logger.info("Analysis complete. Sending response.")
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"Error processing video: {str(e)}")
        return jsonify({'error': str(e)}), 500
        
    finally:
        # Clean up
        if 'filepath' in locals() and os.path.exists(filepath):
            os.remove(filepath)
            logger.info(f"Cleaned up file: {filepath}")

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port) 
