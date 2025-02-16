import os
import numpy as np
import json
import pickle as pkl
from tqdm import tqdm
import pandas as pd

from feature_extraction import Track  
from load_models import load_models 

FEATURES_PATH = 'exports/saved_features.pkl'
EMBEDDINGS_PATH = 'exports/saved_embeddings.pkl'

def predict_and_export_features(dataset_path, save_path=FEATURES_PATH, sample_rate=16000, num_tracks=None, reset_file=False):
    """
    Predict and export features for all tracks in the dataset.

    Parameters:
    dataset_path (str): Path to the dataset directory.
    save_path (str): Path to save the features.
    sample_rate (int): Sample rate for audio processing.
    num_tracks (int): Number of tracks to process. If None, process all tracks.
    reset_file (bool): Whether to reset the save file.

    Returns:
    str: Path to the dataset directory.
    """
    track_paths = get_all_track_paths(dataset_path)
    models = load_models()
    
    if reset_file:
        open(save_path, "w").close()
    
    with open(save_path, 'wb') as f:
        for path in tqdm(track_paths[:num_tracks]):
            track = Track(path=path, sample_rate=sample_rate, models=models)
            track.load_audio()
            track.full_analysis()
            pkl.dump(track.features.__dict__, f)
        
    return dataset_path

def predict_and_export_embeddings(dataset_path, save_path=EMBEDDINGS_PATH, sample_rate=16000, num_tracks=None, reset_file=False):
    """
    Predict and export embeddings for all tracks in the dataset.

    Parameters:
    dataset_path (str): Path to the dataset directory.
    save_path (str): Path to save the embeddings.
    sample_rate (int): Sample rate for audio processing.
    num_tracks (int): Number of tracks to process. If None, process all tracks.
    reset_file (bool): Whether to reset the save file.

    Returns:
    str: Path to the dataset directory.
    """
    track_paths = get_all_track_paths(dataset_path)
    models = load_models()
    
    if reset_file:
        open(save_path, "w").close()
    
    with open(save_path, 'wb') as f:
        for path in tqdm(track_paths[:num_tracks]):
            track = Track(path=path, sample_rate=sample_rate, models=models)
            track.load_audio()
            track.predict_embeddings()
            track.embeddings['path'] = path
            pkl.dump(track.embeddings, f)
        
    return dataset_path

def get_all_track_paths(dataset_path):
    """
    Get all track paths from the dataset directory.

    Parameters:
    dataset_path (str): Path to the dataset directory.

    Returns:
    list: List of track paths.
    """
    paths = []
    for path, _, files in os.walk(dataset_path):
        for file in files:
            fullpath = os.path.join(path, file)
            _, extension = os.path.splitext(fullpath)
            if os.path.isfile(fullpath) and (extension == ".mp3"):
                paths.append(fullpath)
    return paths

def load_features(features_path=FEATURES_PATH):
    """
    Load features from the specified path.

    Parameters:
    features_path (str): Path to the features file.

    Returns:
    pandas.DataFrame: DataFrame containing the loaded features.
    """
    all_features = []
    with open(features_path, 'rb') as f:
        while True:
            try:
                all_features.append(pkl.load(f))
            except EOFError:
                break
        
    return pd.DataFrame(all_features)

def load_embeddings(embeddings_path=EMBEDDINGS_PATH):
    """
    Load embeddings from the specified path.

    Parameters:
    embeddings_path (str): Path to the embeddings file.

    Returns:
    pandas.DataFrame: DataFrame containing the loaded embeddings.
    """
    all_embeddings = []
    with open(embeddings_path, 'rb') as f:
        while True:
            try:
                all_embeddings.append(pkl.load(f))
            except EOFError:
                break
        
    return pd.DataFrame(all_embeddings)