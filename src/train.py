def train_model(model, training_data, epochs, batch_size, model_path):
    model.fit(training_data, training_data, epochs=epochs, batch_size=batch_size, shuffle=False)
    model.save(model_path)
