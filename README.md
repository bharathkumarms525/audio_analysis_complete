# Audio Analysis

This project provides an API for audio transcription, speaker diarization, sentiment analysis, and summarization. It leverages state-of-the-art machine learning models like Whisper for transcription and SpeechBrain for speaker recognition.

## Features

- **Audio Transcription**: Converts audio files into text using OpenAI's Whisper model.
- **Speaker Diarization**: Dynamically detects and labels speakers in the audio using embeddings and clustering.
- **Sentiment Analysis**: Analyzes the sentiment of transcribed text for each speaker segment.
- **Text Summarization**: Generates concise summaries of the transcribed text.
- **Speaker Statistics**: Provides detailed statistics for each speaker, including speaking rate and pitch.

## Project Structure

```
f:\audio
│
├── app1
│   ├── api
│   │   └── routes.py          # FastAPI routes for handling API requests
│   ├── models
│   │   └── response_model.py  # Pydantic models for API responses
│   ├── services
│   │   ├── diarization.py     # Speaker diarization logic
│   │   ├── summarization.py   # Text summarization logic
│   │   ├── transcription.py   # Audio transcription and integration logic
│   │   ├── sentiment.py       # Sentiment analysis logic
│   │   ├── speaker_stats.py   # Speaker statistics calculation
│   └── utils
│       └── audio_utils.py     # Utility functions for audio processing
│
├── requirements.txt           # All the dependencies
└── README.md                  # Project documentation
```

## How It Works

1. **Transcription**:
   - The `transcription.py` service uses Whisper to transcribe audio files into text.
   - Language detection is performed automatically.

2. **Speaker Diarization**:
   - The `diarization.py` service dynamically estimates the number of speakers using silhouette scores.
   - Speaker embeddings are extracted, and clustering is performed to label speakers.

3. **Sentiment Analysis**:
   - The `sentiment.py` service analyzes the sentiment of each speaker's transcribed text.

4. **Summarization**:
   - The `summarization.py` service generates a concise summary of the entire transcription.

5. **Speaker Statistics**:
   - The `speaker_stats.py` service calculates speaking rate and pitch for each speaker.

## API Endpoints

### `/transcribe/` (POST)
- **Description**: Transcribes an audio file and returns detailed information, including speaker diarization, sentiment analysis, and summarization.
- **Request**:
  - `file`: Audio file to be transcribed (e.g., `.mp3`, `.wav`).
- **Response**:
  - `language`: Detected language of the audio.
  - `transcription`: Full transcription of the audio.
  - `segments`: List of speaker segments with start time, end time, speaker label, text, and sentiment.
  - `overall_sentiment`: Overall sentiment of the transcription.
  - `summary`: Summary of the transcription.
  - `speaker_stats`: Statistics for each speaker (e.g., speaking rate, average pitch).

## Installation

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd audio
   ```

2. Create a virtual environment:
   ```bash
   python -m venv myenv
   ```

3. Activate the virtual environment:
   - On Windows:
     ```bash
     myenv\Scripts\activate
     ```
   - On macOS/Linux:
     ```bash
     source myenv/bin/activate
     ```

4. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. Start the FastAPI server:
   ```bash
   uvicorn app1.main:app --reload
   ```

2. Access the API documentation at:
   ```
   http://127.0.0.1:8000/docs
   ```

3. Use the `/transcribe/` endpoint to upload audio files and receive transcription results.

## Dependencies
- fastapi
- uvicorn
- whisper
- torchaudio
- scikit-learn
- speechbrain
- nltk
- numpy
- spacytextblob
- spacy
- nltk
- librosa
- pyloudnorm
- python-multipart

install this in command prompt: python -m spacy download en_core_web_sm


## Future Enhancements

- Add support for real-time audio streaming.
- Enhance speaker diarization with additional clustering techniques.
- Provide more detailed sentiment analysis with advanced NLP models.
