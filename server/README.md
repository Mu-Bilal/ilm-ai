# Course Generation API

This is the backend service for the course generation feature. It provides an API to generate courses by processing URLs and downloading relevant files.

## Setup

1. Make sure you have Python 3.11 installed:
```bash
python3.11 --version
```

2. Create and activate a virtual environment:
```bash
python3.11 -m venv venv
source venv/bin/activate  # On Unix/macOS
# or
.\venv\Scripts\activate  # On Windows
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Run the development server:
```bash
uvicorn app.main:app --reload
```

The API will be available at `http://localhost:8000`

## API Endpoints

### Generate Course
`POST /api/generate-course`

Request body:
```json
{
    "title": "Course Title",
    "description": "Course Description",
    "urls": ["https://example.com/course1", "https://example.com/course2"],
    "topics": ["Topic 1", "Topic 2"]  // Optional
}
```

Response:
```json
{
    "title": "Course Title",
    "description": "Course Description",
    "topics": ["Topic 1", "Topic 2"],
    "files": [
        {
            "name": "document.pdf",
            "url": "path/to/file",
            "type": "document"
        }
    ],
    "progress": 0
}
```

### Health Check
`GET /api/health`

Response:
```json
{
    "status": "healthy"
}
```

## Features

- Asynchronous URL processing
- File downloading and categorization
- Support for various file types (PDF, DOC, PPT, etc.)
- Error handling and logging
- CORS support for frontend integration 