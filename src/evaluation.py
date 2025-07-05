import matplotlib.pyplot as plt
import seaborn as sns

def predict_in_batches(model, data, batch_size=8):
    preds = []
    for i in range(0, len(data), batch_size):
        preds.append(model.predict(data[i:i+batch_size], verbose=0))
    return np.concatenate(preds, axis=0)


def evaluate_model(model, test_data, true_labels):
    reconstructed = predict_in_batches(model, test_data)
    frame_errors = np.mean((test_data - reconstructed) ** 2, axis=(2, 3, 4)).flatten()
    fpr, tpr, _ = roc_curve(true_labels, frame_errors)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:0.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.show()
    print(f"AUC Score: {roc_auc}")


def plot_reconstruction_error(errors, threshold):
    plt.figure()
    plt.plot(errors, label="Reconstruction Error")
    plt.axhline(threshold, color='red', linestyle='--', label="Threshold")
    plt.legend()
    plt.title("Anomaly Detection")
    plt.show()

def plot_error_distributions(train_errors, test_errors, threshold):
    plt.figure()
    sns.histplot(train_errors, color='blue', label='Train', kde=True)
    sns.histplot(test_errors, color='orange', label='Test', kde=True)
    plt.axvline(threshold, color='red', linestyle='--', label='Threshold')
    plt.legend()
    plt.title("Frame-Level Errors")
    plt.show()



