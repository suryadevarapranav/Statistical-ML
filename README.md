# Deepfake Detection Flask Application

This repository contains a Flask application that can classify videos as either original or manipulated (deepfake). The application utilizes a machine learning model to analyze video frames and generate a classification result.

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes.

### Prerequisites

Before you begin, ensure you have the following installed:
- Python 3.6+
- pip (Python package manager)

### Installation

To set up the project environment, follow these steps:

1. Clone the repository to your local machine:
```bash
git clone https://github.com/priyamthakkar2001/SML-Project.git
```

2. Navigate to the project directory:
```bash
cd 'Deepfake Project'
```

3. Install the required Python packages:
```bash
pip install -r requirements.txt
```

## Running the Application

To run the Flask application, use the following command:
```bash
python3 app.py
```
After running the command, you can access the application at `http://127.0.0.1:5000/` in your web browser.

## Using the Application

To use the application:

1. Navigate to the home page at `http://127.0.0.1:5000/`.
2. Click on "Select file" to upload a video file.
3. Click "Upload and Analyze" to submit the file for processing.
4. The application will display the classification results and relevant heatmaps on a new results page.
