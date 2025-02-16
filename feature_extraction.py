import numpy as np
import os
import json
from dataclasses import dataclass, field
from typing import Dict
from pathlib import Path

import essentia
import essentia.standard as es

from IPython.display import Audio
from IPython.core.display import display

GENRE_META_PATH = "pretrained_models/genre_discogs400.json"

@dataclass
class TrackFeatures:
    path: Path = None
    genre: int = None
    instrumental: float = None
    voice: float = None
    valence: float = None
    arousal: float = None
    bpm: float = None
    key: Dict[str, str] = field(default_factory=lambda: ({'temperley': None, 'krumhansl': None, 'edma': None}))
    scale: Dict[str, str] = field(default_factory=lambda: ({'temperley': None, 'krumhansl': None, 'edma': None}))
    loudness: float = None
    danceability: float = None

class Track:
    def __init__(self, path, models=None, sample_rate=16000):
        """
        Initialize the Track object.

        Parameters:
        path (str): Path to the audio file.
        models (dict): Dictionary of pre-trained models.
        sample_rate (int): Sample rate for audio processing.
        """
        self.path = path
        self.fs = sample_rate
        self.audio = None
        self.stereo_audio = None
        self.embeddings = {}
        self.features = TrackFeatures(path=path)
        self.models = models

    def load_audio(self):
        """
        Load the audio file and resample it to the specified sample rate.

        Returns:
        tuple: Stereo audio and sample rate.
        """
        audio, fs, numberChannels, _, _, _ = es.AudioLoader(filename=self.path)()
        mono_audio = es.MonoMixer()(audio.astype(np.float32), numberChannels)
        mono_audio = es.Resample(inputSampleRate=fs, outputSampleRate=self.fs)(mono_audio.astype(np.float32))
        self.audio = mono_audio
        self.stereo_audio = audio
        return self.stereo_audio, fs

    def play_audio(self):
        """
        Play the loaded audio.
        """
        display(Audio(data=self.audio, rate=self.fs, autoplay=True))

    def predict_embeddings(self):
        """
        Predict embeddings using the pre-trained models.

        Returns:
        dict: Dictionary of embeddings.
        """
        self.embeddings['discogs'] = self.models['discogs'](self.audio)
        self.embeddings['musicnn'] = self.models['musicnn'](self.audio)
        return self.embeddings

    def predict_genre(self, meta_data_path=GENRE_META_PATH):
        """
        Predict the genre of the track using the pre-trained model.

        Parameters:
        meta_data_path (str): Path to the genre metadata file.

        Returns:
        str: Predicted genre.
        """
        predictions = self.models['genre'](self.embeddings['discogs'])
        predictions = np.average(predictions, axis=0)
        best_class_index = np.argmax(predictions)

        with open(meta_data_path, 'r') as f:
            meta_data = json.load(f)

        predicted_class = meta_data['classes'][best_class_index]
        self.features.genre = predicted_class
        return predicted_class

    def predict_instrumentalness(self):
        """
        Predict the instrumentalness and voice presence of the track.

        Returns:
        numpy.ndarray: Array containing instrumentalness and voice presence.
        """
        predictions = self.models['instrumental'](np.array(self.embeddings['discogs']))
        prediction = np.average(predictions, axis=0)
        self.features.instrumental = prediction[0]
        self.features.voice = prediction[1]
        return prediction

    def predict_emotions(self):
        """
        Predict the valence and arousal of the track.

        Returns:
        numpy.ndarray: Array containing valence and arousal.
        """
        predictions = self.models['emotion'](self.embeddings['musicnn'])
        prediction = np.average(predictions, axis=0)
        self.features.valence = prediction[0]
        self.features.arousal = prediction[1]
        return prediction

    def predict_rythm(self):
        """
        Predict the BPM (beats per minute) of the track.

        Returns:
        float: BPM of the track.
        """
        self.features.bpm, _, _, _, _ = es.RhythmExtractor2013()(self.audio)
        return self.features.bpm

    def predict_key(self, profile_types=['temperley', 'krumhansl', 'edma']):
        """
        Predict the key and scale of the track using different profile types.

        Parameters:
        profile_types (list): List of profile types to use for key prediction.

        Returns:
        tuple: Dictionary of keys and scales for each profile type.
        """
        for profile_type in profile_types:
            self.features.key[profile_type], self.features.scale[profile_type], _ = es.KeyExtractor(
                sampleRate=self.fs,
                profileType=profile_type
            )(self.audio)
        return self.features.key, self.features.scale

    def predict_loudness(self):
        """
        Predict the loudness of the track.

        Returns:
        float: Loudness of the track.
        """
        _, _, self.features.loudness, _ = es.LoudnessEBUR128(sampleRate=self.fs)(self.stereo_audio)
        return self.features.loudness

    def predict_danceability(self):
        """
        Predict the danceability of the track.

        Returns:
        float: Danceability of the track.
        """
        self.features.danceability, _ = es.Danceability(sampleRate=self.fs)(self.audio)
        return self.features.danceability

    def full_analysis(self):
        """
        Perform a full analysis of the track, predicting all features.

        Returns:
        TrackFeatures: Object containing all predicted features.
        """
        self.predict_embeddings()
        self.predict_genre()
        self.predict_instrumentalness()
        self.predict_emotions()
        self.predict_rythm()
        self.predict_key()
        self.predict_loudness()
        self.predict_danceability()
        return self.features