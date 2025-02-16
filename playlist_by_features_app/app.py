import sys
import numpy as np
import streamlit as st
import pandas as pd

sys.path.append('../')
from load_dataset import load_features
from feature_extraction import Track

from multiprocessing import Event

FEATURES_PATH = '../exports/saved_features.pkl'
USED_MODEL_FOR_KEY = 'krumhansl'

# Event to trigger export
make_export_event = Event()

def load_analysis():
    """
    Load and preprocess the analysis data.

    Returns:
    pandas.DataFrame: DataFrame containing the preprocessed analysis data.
    """
    df_features = load_features(FEATURES_PATH)
    df_features["genre_big"] = df_features["genre"].str.split("---").str[0]
    df_features["genre_small"] = df_features["genre"].str.split("---").str[1]
    df_features['is_it_instrumental'] = df_features['instrumental'] > 0.5
    df_features[USED_MODEL_FOR_KEY+'_key'] = df_features['key'].apply(lambda x: x[USED_MODEL_FOR_KEY])
    df_features[USED_MODEL_FOR_KEY+'_scale'] = df_features['scale'].apply(lambda x: x[USED_MODEL_FOR_KEY])
    return df_features

# Load all tracks and preprocess the data
all_tracks = load_analysis()
selected_tracks = all_tracks

# Sidebar content
st.sidebar.write(f'Using analysis data from `{FEATURES_PATH}`.')
st.sidebar.write('Loaded audio analysis for', len(all_tracks), 'tracks.')

# Create tabs for genres and other filters
genres, other = st.tabs(["Genres", "Other"])

# Display genre distribution
genres.write('#### Main')
genres.bar_chart(pd.DataFrame(all_tracks["genre_big"].value_counts()))

# Genre selection
style_select = genres.pills('Select one or more', all_tracks["genre_big"].unique(), selection_mode='multi')
if style_select:
    genres.write('#### Sub')
    
    selected_tracks = all_tracks[
        all_tracks['genre_big'].str.contains('|'.join(style_select))
    ].sort_values(by="genre_small")
    genres.bar_chart(pd.DataFrame(selected_tracks["genre_small"].value_counts()))
    sub_style_select = genres.pills('', selected_tracks["genre_small"].unique(), selection_mode='multi')
    if sub_style_select:
        selected_tracks = selected_tracks[
            selected_tracks['genre_small'].str.contains('|'.join(sub_style_select))
        ]
        make_export_event.set()

# Tempo filter
other.write('## Tempo')
min_bpm = np.round(selected_tracks['bpm'].min())
max_bpm = np.round(selected_tracks['bpm'].max())
bpm_options = list(range(int(min_bpm), int(max_bpm)+1))
start_bpm, end_bpm = other.select_slider(
    "Select a range of BPM",
    options=bpm_options,
    value=(min_bpm, max_bpm),
)
if start_bpm or end_bpm:
    selected_tracks = selected_tracks.loc[selected_tracks['bpm'] >= start_bpm]
    selected_tracks = selected_tracks.loc[selected_tracks['bpm'] <= end_bpm]
    make_export_event.set()

# Danceability filter
other.write('## Danceability')
dance_options = list(np.around(np.arange(0, 3.01, 0.01), decimals=2))
start_dance, end_dance = other.select_slider(
    "Select a range of values",
    options=dance_options,
    value=(0, max(dance_options)),
)
if start_dance or end_dance:
    selected_tracks = selected_tracks.loc[selected_tracks['danceability'] >= start_dance]
    selected_tracks = selected_tracks.loc[selected_tracks['danceability'] <= end_dance]
    make_export_event.set()

# Instrumental/Vocal filter
other.write('## Instrumental/Vocal')  
inst_options = ['Instrumental', 'Vocal', 'All']
instrumental_selection = other.segmented_control(
    "", inst_options,
)
if instrumental_selection:
    if instrumental_selection == inst_options[0]:
        selected_tracks = selected_tracks.loc[selected_tracks['is_it_instrumental'] == True]
    elif instrumental_selection == inst_options[1]:
        selected_tracks = selected_tracks.loc[selected_tracks['is_it_instrumental'] == False]
    make_export_event.set()

# Arousal and Valence filters
other.write('## Arousal and Valence') 

arousal_options = np.around(selected_tracks['arousal'].unique(), decimals=2)
arousal_options = list(np.sort(arousal_options))
start_arousal, end_arousal = other.select_slider(
    "Select a range of Arousal",
    options=arousal_options,
    value=(min(arousal_options), max(arousal_options)),
)
if start_arousal or end_arousal:
    selected_tracks = selected_tracks.loc[selected_tracks['arousal'] >= start_arousal]
    selected_tracks = selected_tracks.loc[selected_tracks['arousal'] <= end_arousal]
    make_export_event.set()

valence_options = np.around(selected_tracks['valence'].unique(), decimals=2)
valence_options = list(np.sort(valence_options))
start_valence, end_valence = other.select_slider(
    "Select a range of Valence",
    options=valence_options,
    value=(min(valence_options), max(valence_options)),
)
if start_valence or end_valence:
    selected_tracks = selected_tracks.loc[selected_tracks['valence'] >= start_valence]
    selected_tracks = selected_tracks.loc[selected_tracks['valence'] <= end_valence]
    make_export_event.set()

# Key-Scale filter
other.write('## Key-Scale')  
key_options = selected_tracks.sort_values(by=USED_MODEL_FOR_KEY+'_key')
key_options = key_options[USED_MODEL_FOR_KEY+'_key'].unique()
key_select = other.pills('', key_options, selection_mode='multi') 

if key_select:
    if len(key_select) != None:
        selected_tracks = selected_tracks[
            selected_tracks[USED_MODEL_FOR_KEY+'_key'].str.contains('|'.join(key_select))
        ]
    make_export_event.set()
    
scale_options = ['major','minor']
scale_select = other.pills('', scale_options, selection_mode='multi')
if scale_select:
    selected_tracks = selected_tracks[
        selected_tracks[USED_MODEL_FOR_KEY+'_scale'].str.contains('|'.join(scale_select))
    ]
    make_export_event.set()

# Display the generated playlist
if make_export_event.is_set:
    st.write('---')
    st.write('## Generated Playlist')
    max_tracks = st.number_input('Maximum number of tracks (0 for all):', value=10)
    st.toast(f'Number of Filtered Tracks: {len(selected_tracks)}')
    selected_tracks = selected_tracks.sample(frac=1)
    st.dataframe(selected_tracks[:max_tracks])
    for path in selected_tracks['path'][:max_tracks]:
        path = '../'+path
        audio, fs = Track(path).load_audio()
        st.audio(audio.T, sample_rate=fs, format="audio/mp3")  
        make_export_event.clear()