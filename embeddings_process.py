from load_dataset import *
import numpy as np

def cosine_similarity(vec1, vec2):
    """
    Calculate the cosine similarity between two vectors.

    Parameters:
    vec1 (numpy.ndarray): The first vector.
    vec2 (numpy.ndarray): The second vector.

    Returns:
    float: The cosine similarity between vec1 and vec2.
    """
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

def calc_similarity(tracks, chosen_track, dataset_names=['discogs', 'musicnn']):
    """
    Calculate the cosine similarity between the chosen track and all other tracks
    for the specified dataset names.

    Parameters:
    tracks (pandas.DataFrame): DataFrame containing the tracks and their embeddings.
    chosen_track (pandas.DataFrame): DataFrame containing the chosen track and its embeddings.
    dataset_names (list): List of dataset names to calculate similarity for.

    Returns:
    pandas.DataFrame: DataFrame with added columns for cosine similarity for each dataset.
    """
    for dataset_name in dataset_names:
        # Extract the embedding of the chosen track for the current dataset
        chosen_embedding = chosen_track.iloc[0][dataset_name]
        
        # Calculate the cosine similarity between the chosen track and all other tracks
        tracks['cosine_similarity_' + dataset_name] = tracks[dataset_name].apply(
            lambda x: cosine_similarity(chosen_embedding, x.T)
        )
    return tracks