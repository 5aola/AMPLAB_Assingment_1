import essentia
from essentia.standard import TensorflowPredictEffnetDiscogs, TensorflowPredictMusiCNN, TensorflowPredict2D

# Paths to the pre-trained model files
DISCOGS_PATH = "pretrained_models/discogs-effnet-bs64-1.pb"
MUSICNN_PATH = "pretrained_models/msd-musicnn-1.pb"
GENRE_MODEL_PATH = "pretrained_models/genre_discogs400-discogs-effnet-1.pb"
INSTRUMENTAL_MODEL_PATH = "pretrained_models/voice_instrumental-discogs-effnet-1.pb"
EMOTIONS_MODEL_PATH = "pretrained_models/emomusic-msd-musicnn-2.pb"

def load_models():
    """
    Load the pre-trained models for various tasks.

    Returns:
    dict: Dictionary containing the loaded models.
    """
    models = {}
    
    models['discogs'] = TensorflowPredictEffnetDiscogs(
        graphFilename=DISCOGS_PATH, 
        output="PartitionedCall:1"
    )
    
    models['musicnn'] = TensorflowPredictMusiCNN(
        graphFilename=MUSICNN_PATH, 
        output="model/dense/BiasAdd"
    )

    models['genre'] = TensorflowPredict2D(
        graphFilename=GENRE_MODEL_PATH, 
        input="serving_default_model_Placeholder", 
        output="PartitionedCall:0"
    )
    
    models['instrumental'] = TensorflowPredict2D(
        graphFilename=INSTRUMENTAL_MODEL_PATH, 
        output="model/Softmax"
    )
    
    models['emotion'] = TensorflowPredict2D(
        graphFilename=EMOTIONS_MODEL_PATH, 
        output="model/Identity"
    )
    
    return models