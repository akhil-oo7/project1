from flask import Flask, render_template, request, jsonify
import os
import logging
import base64
from werkzeug.utils import secure_filename
from video_processor import VideoProcessor
from content_moderator import ContentModerator
import cv2

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100MB max file size

# Ensure upload directory exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Initialize components with trained model
try:
    video_processor = VideoProcessor()
    content_moderator = ContentModerator(train_mode=False)  # Use trained model
    logger.info("Successfully initialized video processor and content moderator")
except Exception as e:
    logger.error(f"Failed to initialize components: {str(e)}")
    raise

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze_video():
    if 'video' not in request.files:
        logger.error("No video file provided in request")
        return jsonify({'error': 'No video file provided'}), 400
    
    file = request.files['video']
    if file.filename == '':
        logger.error("Empty filename provided")
        return jsonify({'error': 'No selected file'}), 400
    
    if file:
        try:
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            logger.info(f"Saving file to: {filepath}")
            file.save(filepath)
            
            # Process video and get frames
            logger.info("Starting frame extraction")
            frames = video_processor.extract_frames(filepath)
            logger.info(f"Extracted {len(frames)} frames")
            
            # Analyze frames for content moderation
            logger.info("Starting content analysis")
            results = content_moderator.analyze_frames(frames)
            logger.info("Content analysis completed")
            
            # Calculate overall video safety
            unsafe_frames = [r for r in results if r['flagged']]
            total_frames = len(results)
            unsafe_percentage = (len(unsafe_frames) / total_frames) * 100 if total_frames > 0 else 0
            
            # Prepare response with frame images
            response = {
                'status': 'UNSAFE' if unsafe_frames else 'SAFE',
                'total_frames': total_frames,
                'unsafe_frames': len(unsafe_frames),
                'unsafe_percentage': unsafe_percentage,
                'confidence': 1.0 if not unsafe_frames else max(r['confidence'] for r in unsafe_frames),
                'details': []
            }
            
            if unsafe_frames:
                for frame_idx, result in enumerate(results):
                    if result['flagged']:
                        # Convert frame to base64 for display
                        frame = frames[frame_idx]
                        _, buffer = cv2.imencode('.jpg', frame)
                        frame_base64 = base64.b64encode(buffer).decode('utf-8')
                        
                        response['details'].append({
                            'frame': frame_idx,
                            'reason': result['reason'],
                            'confidence': result['confidence'],
                            'image': frame_base64
                        })
            
            # Clean up uploaded file
            logger.info("Cleaning up uploaded file")
            os.remove(filepath)
            
            return jsonify(response)
            
        except Exception as e:
            logger.error(f"Error during video analysis: {str(e)}", exc_info=True)
            # Clean up file if it exists
            if 'filepath' in locals() and os.path.exists(filepath):
                try:
                    os.remove(filepath)
                except:
                    pass
            return jsonify({'error': f'An error occurred while analyzing the video: {str(e)}'}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 10000))
    app.run(host='0.0.0.0', port=port) 
