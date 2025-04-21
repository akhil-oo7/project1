from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
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
CORS(app)  # Enable CORS for all routes

# Create uploads directory if it doesn't exist
UPLOAD_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
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
        return jsonify({
            "status": "error",
            "message": "Server components not properly initialized"
        }), 500

    if 'video' not in request.files:
        return jsonify({
            "status": "error",
            "message": "No video file provided"
        }), 400
    
    file = request.files['video']
    if file.filename == '':
        return jsonify({
            "status": "error",
            "message": "No file selected"
        }), 400
    
    # Validate file type
    if not file.filename.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
        return jsonify({
            "status": "error",
            "message": "Unsupported file format. Please upload MP4, AVI, MOV, or MKV."
        }), 400
    
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
                return jsonify({
                    "status": "error",
                    "message": "No frames could be extracted from the video"
                }), 400
            logger.info(f"Extracted {len(frames)} frames")
        except Exception as e:
            logger.error(f"Error processing video: {str(e)}")
            return jsonify({
                "status": "error",
                "message": f"Error processing video: {str(e)}"
            }), 500
        
        # Analyze frames
        try:
            results = content_moderator.analyze_frames(frames)
            if not results:
                return jsonify({
                    "status": "error",
                    "message": "No analysis results returned"
                }), 500
            logger.info(f"Analyzed {len(results)} frames")
        except Exception as e:
            logger.error(f"Error analyzing frames: {str(e)}")
            return jsonify({
                "status": "error",
                "message": f"Error analyzing frames: {str(e)}"
            }), 500
        
        # Calculate percentages
        total_frames = len(results)
        unsafe_frames = [r for r in results if r['flagged']]
        unsafe_percentage = (len(unsafe_frames) / total_frames) * 100 if total_frames > 0 else 0
        
        # Prepare response
        response = {
            "status": "success",
            "total_frames": total_frames,
            "frame_results": []
        }
        
        # Process frames and add images
        for i, result in enumerate(results):
            if result['flagged']:
                try:
                    # Convert frame to base64 for display
                    _, buffer = cv2.imencode('.jpg', frames[i])
                    frame_base64 = base64.b64encode(buffer).decode('utf-8')
                    
                    response['frame_results'].append({
                        'frame_index': i,
                        'reason': result['reason'],
                        'confidence': result['confidence'],
                        'image': frame_base64
                    })
                except Exception as e:
                    logger.warning(f"Error processing frame {i}: {str(e)}")
                    continue
        
        if unsafe_frames:
            response['unsafe_percentage'] = unsafe_percentage
            response['safe_percentage'] = 100 - unsafe_percentage
        else:
            response['safe_percentage'] = 100.0
            response['unsafe_percentage'] = 0.0
        
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"An unexpected error occurred: {str(e)}")
        return jsonify({
            "status": "error",
            "message": f"An unexpected error occurred: {str(e)}"
        }), 500
        
    finally:
        # Clean up
        if 'filepath' in locals() and os.path.exists(filepath):
            try:
                os.remove(filepath)
                logger.info(f"Cleaned up file: {filepath}")
            except Exception as e:
                logger.warning(f"Error cleaning up file: {str(e)}")

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 10000))
    app.run(host='0.0.0.0', port=port) 
