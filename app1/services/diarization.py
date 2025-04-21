import os
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import torchaudio
from app1.services.sentiment import ensemble_sentiment
from app1.utils.audio_utils import get_speaker_embedding, whisper_model

def estimate_num_speakers(embeddings, min_clusters=2, max_clusters=10):
    """
    Estimate the number of speakers dynamically using silhouette score.
    """
    best_k = min_clusters
    best_score = -1
    n_samples = len(embeddings)
    max_possible = min(max_clusters, n_samples - 1)

    for k in range(min_clusters, max_possible + 1):
        kmeans = KMeans(n_clusters=k, random_state=0)
        labels = kmeans.fit_predict(embeddings)
        if len(np.unique(labels)) == 1:  # Skip if only one cluster is found
            continue
        score = silhouette_score(embeddings, labels)
        if score > best_score:
            best_score = score
            best_k = k
    return best_k

def diarize_with_speechbrain(audio_path, dynamic_speakers=True, min_clusters=2, max_clusters=10):
    """
    Perform speaker diarization with optional dynamic speaker detection.
    """
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"Audio file not found at {audio_path}")

    # Step 1: Transcribe with Whisper
    result = whisper_model.transcribe(audio_path)
    segments = result["segments"]

    # Step 2: Load the full audio
    waveform, sample_rate = torchaudio.load(audio_path)

    # Step 3: Extract embeddings for each segment
    embeddings = []
    for seg in segments:
        start_sample = int(seg['start'] * sample_rate)
        end_sample = int(seg['end'] * sample_rate)
        segment_audio = waveform[:, start_sample:end_sample]
        emb = get_speaker_embedding(segment_audio, sample_rate)
        embeddings.append(emb)
    embeddings = np.vstack(embeddings)

    # Step 4: Estimate number of speakers if dynamic_speakers is True
    if dynamic_speakers:
        num_speakers = estimate_num_speakers(embeddings, min_clusters, max_clusters)
    else:
        num_speakers = min_clusters

    # Step 5: Cluster embeddings into the determined number of speakers
    labels = KMeans(n_clusters=num_speakers, random_state=0).fit_predict(embeddings)

    # Step 6: Assign speaker labels and sentiments
    diarized_segments = []
    for i, seg in enumerate(segments):
        speaker = f"Speaker_{labels[i]}"
        text = seg["text"]
        combined_sentiment = ensemble_sentiment(text)

        diarized_segments.append({
            "start": seg["start"],
            "end": seg["end"],
            "speaker": speaker,
            "text": text,
            "sentiment": combined_sentiment
        })

    return diarized_segments

