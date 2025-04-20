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
try:
    video_processor = VideoProcessor()
    logger.info("VideoProcessor initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize VideoProcessor: {str(e)}")
    logger.error(f"Traceback: {traceback.format_exc()}")
    video_processor = None

try:
    content_moderator = ContentModerator()
    logger.info("ContentModerator initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize ContentModerator: {str(e)}")
    logger.error(f"Traceback: {traceback.format_exc()}")
    content_moderator = None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze_video():
    if not video_processor or not content_moderator:
        error_msg = "Server components not properly initialized"
        logger.error(error_msg)
        return jsonify({'error': error_msg}), 500

    if 'video' not in request.files:
        logger.error("No video file in request")
        return jsonify({'error': 'No video file provided'}), 400
    
    file = request.files['video']
    if file.filename == '':
        logger.error("Empty filename")
        return jsonify({'error': 'No file selected'}), 400
    
    # Validate file type
    if not file.filename.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
        error_msg = "Unsupported file format. Please upload MP4, AVI, MOV, or MKV."
        logger.error(error_msg)
        return jsonify({'error': error_msg}), 400
    
    try:
        # Save the uploaded file
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        logger.info(f"File saved to {filepath}")
        
        # Process video
        try:
            frames = video_processor.extract_frames(filepath)
            if not frames:
                error_msg = "No frames could be extracted from the video"
                logger.error(error_msg)
                return jsonify({'error': error_msg}), 400
            logger.info(f"Extracted {len(frames)} frames")
        except Exception as e:
            error_msg = f"Error processing video: {str(e)}"
            logger.error(error_msg)
            logger.error(f"Traceback: {traceback.format_exc()}")
            return jsonify({'error': error_msg}), 500
        
        # Analyze frames
        try:
            results = content_moderator.analyze_frames(frames)
            if not results:
                error_msg = "No analysis results returned"
                logger.error(error_msg)
                return jsonify({'error': error_msg}), 500
            logger.info(f"Analyzed {len(results)} frames")
        except Exception as e:
            error_msg = f"Error analyzing frames: {str(e)}"
            logger.error(error_msg)
            logger.error(f"Traceback: {traceback.format_exc()}")
            return jsonify({'error': error_msg}), 500
        
        # Prepare response
        flagged_frames = []
        for i, result in enumerate(results):
            if result['flagged']:
                try:
                    # Convert frame to base64 for display
                    _, buffer = cv2.imencode('.jpg', frames[i])
                    frame_base64 = base64.b64encode(buffer).decode('utf-8')
                    
                    flagged_frames.append({
                        'frame_index': i,
                        'reason': result['reason'],
                        'confidence': result['confidence'],
                        'image': frame_base64
                    })
                except Exception as e:
                    logger.warning(f"Error processing frame {i}: {str(e)}")
                    continue
        
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
        
        logger.info(f"Analysis complete. Status: {response['status']}, Unsafe Percentage: {unsafe_percentage:.2f}%")
        return jsonify(response)
        
    except Exception as e:
        error_msg = f"An unexpected error occurred: {str(e)}"
        logger.error(error_msg)
        logger.error(f"Traceback: {traceback.format_exc()}")
        return jsonify({'error': error_msg}), 500
        
    finally:
        # Clean up
        if 'filepath' in locals() and os.path.exists(filepath):
            try:
                os.remove(filepath)
                logger.info(f"Cleaned up file: {filepath}")
            except Exception as e:
                logger.warning(f"Error cleaning up file: {str(e)}")

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port) 
