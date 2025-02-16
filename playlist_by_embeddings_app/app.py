import sys
import numpy as np
import streamlit as st

sys.path.append('../')
from load_dataset import load_embeddings 
from feature_extraction import Track
from embeddings_process import *

EMBEDDINGS_PATH = '../exports/saved_embeddings.pkl'

def load_track_embeddings():
    """
    Load track embeddings from the specified path and average them.

    Returns:
    pandas.DataFrame: DataFrame containing the loaded and averaged embeddings.
    """
    track_embeddings = load_embeddings(EMBEDDINGS_PATH)
    track_embeddings['discogs'] = track_embeddings['discogs'].apply(lambda x: np.average(x, axis=0))
    track_embeddings['musicnn'] = track_embeddings['musicnn'].apply(lambda x: np.average(x, axis=0))
    return track_embeddings

# Load all track embeddings
all_tracks = load_track_embeddings()

# Sidebar content
st.sidebar.write(f'Using analysis data from `{EMBEDDINGS_PATH}`.')
st.sidebar.write('Loaded audio analysis for', len(all_tracks), 'tracks.')

# Main content
st.write('# Audio Playlist Generator using Embeddings')

# Button to choose a random track
pressed = st.button("Choose a Random Track", type="primary")
if pressed:
    # Select a random track from the dataset
    chosen_track = all_tracks.sample(n=1)
    path = chosen_track['path']
    path = '../' + path
    st.write(path)
    
    # Load and play the chosen track
    audio, fs = Track(path.values[0]).load_audio()
    st.audio(audio.T, sample_rate=fs, format="audio/mp3")

st.write('---')
st.write('## Most Similar Tracks')

# Create two columns for displaying similar tracks
left_col, right_col = st.columns(2)
left_col.write('### effnet-discogs')
right_col.write('### msd-musicnn')

if pressed:
    # Calculate similarity between the chosen track and all other tracks
    calculated_dist = calc_similarity(all_tracks, chosen_track)
    
    # Sort tracks by similarity using effnet-discogs embeddings
    sorted_tracks = calculated_dist.sort_values(by='cosine_similarity_discogs', ascending=False)
    for path, similarity in zip(sorted_tracks['path'][1:11], sorted_tracks['cosine_similarity_discogs'][1:11]):
        path = '../' + path
        similarity = np.around(similarity * 100, decimals=2)
        left_col.write(f"Similarity: {similarity}%")
        
        # Load and play the similar track
        audio, fs = Track(path).load_audio()
        left_col.audio(audio.T, sample_rate=fs, format="audio/mp3")
    
    # Sort tracks by similarity using msd-musicnn embeddings
    sorted_tracks = calculated_dist.sort_values(by='cosine_similarity_musicnn', ascending=False)
    for path, similarity in zip(sorted_tracks['path'][1:11], sorted_tracks['cosine_similarity_musicnn'][1:11]):
        path = '../' + path
        similarity = np.around(similarity * 100, decimals=2)
        right_col.write(f"Similarity: {similarity}%")
        
        # Load and play the similar track
        audio, fs = Track(path).load_audio()
        right_col.audio(audio.T, sample_rate=fs, format="audio/mp3")