# Video Content Moderation System

A deep learning-based system for analyzing videos and detecting inappropriate content.

## Features
- Video frame analysis using ResNet-50
- Real-time content moderation
- Web interface for easy video upload and analysis
- Confidence scoring for detected content

## Local Setup
1. Clone the repository:
```bash
git clone [your-repository-url]
cd video-content-moderation
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Start the web server:
```bash
python app.py
```

5. Access the web interface at `http://localhost:5000`

## Deployment on Render
1. Fork this repository to your GitHub account
2. Sign up at [Render](https://render.com)
3. Create a new Web Service
4. Connect your GitHub repository
5. Configure the service:
   - Name: video-content-moderation
   - Environment: Python
   - Build Command: `pip install -r requirements.txt`
   - Start Command: `gunicorn app:app`
6. Click "Create Web Service"

The app will be automatically deployed and available at your Render URL.

## Project Structure
```
├── app.py                 # Flask web server
├── content_moderator.py   # Core content analysis
├── train.py              # Model training
├── video_processor.py    # Video processing
├── requirements.txt      # Dependencies
└── templates/           # Web interface
    └── index.html
```

## Requirements
- Python 3.7+
- PyTorch
- Transformers
- Flask
- OpenCV
- Gunicorn (for production)

## Notes
- The model is hosted on Hugging Face
- Free tier on Render has limitations on processing power
- Consider upgrading Render plan for better performance

## License
MIT License

## Contributing
[Your contribution guidelines] 