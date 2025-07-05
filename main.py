from src.config import *
from src.data_loader import preprocess_clips, generate_frame_labels
from src.model import build_model
from src.train import train_model
from src.evaluation import evaluate_model, plot_error_distributions, plot_reconstruction_error, predict_in_batches
from tensorflow.keras.models import load_model
import os
import numpy as np

def run_pipeline():
    MODEL_PATH = "model/model.hdf5" 

    training_data = preprocess_clips(TRAIN_DIR, IMG_SIZE)

    if os.path.exists(MODEL_PATH):
        print(f"Model found at {MODEL_PATH}. Loading pre-trained model.")
        model = load_model(MODEL_PATH, compile=False)
    else:
        print(f"Model not found at {MODEL_PATH}. Starting training...")
        # If model doesn't exist, run the training process
        training_data = preprocess_clips(TRAIN_DIR, IMG_SIZE)
        model = build_model()
        model.fit(training_data, training_data, epochs=EPOCHS, batch_size=BATCH_SIZE, shuffle=False)
        
        print(f"Training complete. Saving model to {MODEL_PATH}")
        model.save(MODEL_PATH)

    train_reconstructed = predict_in_batches(model, training_data)
    train_errors = np.mean((training_data - train_reconstructed)**2, axis=(2, 3, 4)).flatten()
    threshold = np.mean(train_errors) + 2 * np.std(train_errors)

    test_data_all = preprocess_clips(TEST_DIR, IMG_SIZE)
    test_reconstructed_all = predict_in_batches(model, test_data_all)
    test_errors_all = np.mean((test_data_all - test_reconstructed_all)**2, axis=(2,3,4)).flatten()

    plot_reconstruction_error(test_errors_all, threshold)
    plot_error_distributions(train_errors, test_errors_all, threshold)

    test_data_gt = preprocess_clips(TEST_DIR, IMG_SIZE, filter_with_gt=True)
    labels = generate_frame_labels(TEST_DIR)
    evaluate_model(model, test_data_gt, labels)

if __name__ == "__main__":
    run_pipeline()
