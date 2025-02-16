import essentia
from load_dataset import predict_and_export_features, predict_and_export_embeddings

DATASET = 'audio_chunks'

if __name__ == "__main__":
    essentia.log.warningActive = False 

    predict_and_export_features(DATASET, num_tracks=None, reset_file=False)
    predict_and_export_embeddings(DATASET, num_tracks=None, reset_file=False)