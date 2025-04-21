# Content Moderation API

A Flask-based API for content moderation using machine learning models.

## Features

- Video content analysis
- Frame-by-frame moderation
- Safety percentage calculation
- RESTful API endpoints

## Setup

1. Clone the repository:
```bash
git clone https://github.com/akhil-oo7/project1.git
cd project1
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the application:
```bash
python app.py
```

## API Endpoints

- `GET /`: Welcome message
- `GET /api/health`: Health check endpoint
- `POST /analyze`: Analyze video content (coming soon)

## Deployment

This application is configured for deployment on Render.com. The following environment variables can be set:

- `PORT`: Server port (default: 10000)

## License

MIT License 