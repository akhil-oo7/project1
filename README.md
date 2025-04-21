# Video Content Moderation

A Flask web application that analyzes videos for inappropriate content using machine learning.

## Features

- Upload and analyze videos for inappropriate content
- Real-time frame analysis
- Safe/Unsafe content percentage calculation
- Detailed frame-by-frame analysis for unsafe content

## Local Setup

1. Clone the repository:
```bash
git clone <your-repo-url>
cd video-content-moderator
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Run the application:
```bash
python app.py
```

5. Open your browser and navigate to `http://localhost:5000`

## Deployment on Render

1. Create a new Web Service on Render
2. Connect your GitHub repository
3. Configure the service:
   - Environment: Python
   - Build Command: `pip install -r requirements.txt`
   - Start Command: `gunicorn app:app`
   - Add environment variables:
     - `PYTHON_VERSION`: 3.10.0
     - `PORT`: 10000

## Project Structure

```
.
├── app.py              # Main Flask application
├── content_moderator.py # Content moderation logic
├── video_processor.py  # Video processing utilities
├── requirements.txt    # Python dependencies
├── render.yaml         # Render deployment configuration
└── templates/          # HTML templates
    └── index.html      # Main web interface
```

## License

MIT

## Contributing
[Your contribution guidelines] 