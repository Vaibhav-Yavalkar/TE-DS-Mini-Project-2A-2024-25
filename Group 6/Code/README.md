# YouTube Sentiment Analyser and Title Generator

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](https://github.com/TejasDeshmukh13/Yt_sentiment-title-generation/blob/main/LICENSE)

This project is a web-based application that provides sentiment analysis of YouTube video comments and generates suitable video titles using Natural Language Processing (NLP). It's built using **Flask**, **YouTube Data API**, **Hugging Face Transformers**, and **Python**.

## Features

- **YouTube Video Sentiment Analysis**: Analyze the sentiment (positive, negative, neutral) of YouTube video comments.
- **Title Recommendation**: Generate optimized video titles using NLP models.
- **Transcript Summarization**: Automatically summarize the transcript of YouTube videos to extract key points.
- **Channel Statistics**: Display detailed statistics about YouTube channels, such as total views, subscriber count, and more.
- **Real-time Data**: Fetch data directly from YouTube via the YouTube Data API, ensuring up-to-date statistics and information.

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [Features](#features)
- [Project Structure](#project-structure)
- [Contributors](#contributors)
- [License](#license)

## Installation

### Prerequisites

- Python 3.x
- Flask
- MySQL
- YouTube Data API Key
- Hugging Face Transformers
- Jinja2 for templating
- Chart.js for visualizing data

### Setup

1. Clone the repository:

```bash
git clone https://github.com/TejasDeshmukh13/Yt_sentiment-title-generation.git
cd Yt_sentiment-title-generation
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up environment variables for your API keys and other configuration settings:
```bash
export YOUTUBE_API_KEY='your_api_key_here'
```

4. Run the Flask app:
```bash
python app.py
```

5. Visit the app in your browser:
```bash
http://127.0.0.1:5000/
```

## Usage

### Sentiment Analysis

- Enter the URL of a YouTube video to analyze its comment sentiments.<br>
- The app will fetch the video comments and display a sentiment distribution chart (Positive, Negative, Neutral).

### Title Generation

- For the same YouTube video, the app will recommend suitable titles based on the content of the video.

### Transcript Summarization

- Input the video link, and the app will fetch and summarize the video's transcript using NLP techniques, giving a concise overview of the content.

## Project Structure

```bash
Yt_sentiment-title-generation/
├── app.py                    # Main Flask application file
├── channel.py                # Channel statistics logic
├── channel_data.py           # Additional channel data processing logic
├── trainmodel.py             # Model training script
├── requirements.txt          # Dependencies for the project
├── templates/                # Folder for HTML templates
│   ├── forgot_password.html   # HTML for password recovery
│   ├── index.html            # Main landing page
│   ├── landing_page.html     # Landing page for the app
│   ├── login.html            # User login page
│   ├── page1.html            # Additional page 1
│   ├── page2.html            # Additional page 2
│   ├── profile.html          # User profile page
│   ├── res.html              # HTML for displaying results
│   ├── reset_password.html    # HTML for resetting passwords
│   ├── senti_result.html      # Results page for sentiment analysis
│   ├── sentiment.html         # Sentiment analysis input page
│   ├── signup.html           # User signup page
│   ├── summary.html          # Summary of results
│   ├── titlepage.html        # Title generation page
│   └── link.html             # HTML for link submission
├── static/                   # Folder for static files
│   ├── css/                  # CSS files
│   │   ├── landing_page.css   # CSS for landing page
│   │   ├── login.css          # CSS for login page
│   │   ├── page1.css          # CSS for dashboard
│   │   ├── page2.css          # CSS for channel stats
│   │   ├── profile.css        # CSS for profile page
│   │   ├── res.css            # CSS for results page
│   │   ├── signup.css         # CSS for signup page
│   │   └── styles.css         # Main stylesheet
│   ├── js/                   # JavaScript files
│   │   ├── landing_page.js    # JavaScript for landing page
│   │   ├── login.js           # JavaScript for login page
│   │   ├── page1.js           # JavaScript for dahsboard
│   │   ├── page2.js           # JavaScript for channel stats
│   │   ├── profile.js         # JavaScript for profile page
│   │   ├── res.js             # JavaScript for results page
│   │   └── scripts.js         # Main script file
│   └── images/               # Image files 
│       ├── Brochure Zephyr-1.pdf_compressed.pdf # Brochure PDF
│       ├── Demo Video.mp4     # Demo video of the project
│       ├── F1.png             # Example image
│       ├── channel.jpg        # Channel image
│       ├── image.png          # Other image file
│       ├── sentiment.jpg      # Sentiment analysis image
│       ├── sentiment_pie_chart.png # Pie chart for sentiment
│       ├── summarizer.png     # Image for summarization
│       ├── title.png          # Title generation image
│       └── other_image.png    # Another image file
├── utils/                    # Utility functions
│   └── helpers.py            # Helper functions for the app
└── README.md                 # Project documentation
```

## Contributors
<ul>
<a href="https://github.com/kisanjena">KisanKumar Jena</a><br>
<a href="https://github.com/sakshe27">Sakshi Kadam</a><br>
<a href="https://github.com/PriyankaB26">Priyanka Barman</a><br>
<a href="https://github.com/TejasDeshmukh13">Tejas Deshmukh</a>
</ul>
