def build_model():
    model = Sequential()
    # Encoder
    model.add(TimeDistributed(Conv2D(128, (5, 5), strides=4, padding='same'), input_shape=(10, IMG_SIZE, IMG_SIZE, 1)))
    model.add(LayerNormalization())
    model.add(TimeDistributed(Conv2D(64, (3, 3), strides=2, padding='same')))
    model.add(LayerNormalization())

    # Bottleneck
    model.add(ConvLSTM2D(64, (3, 3), padding="same", return_sequences=True))
    model.add(LayerNormalization())
    model.add(ConvLSTM2D(32, (3, 3), padding="same", return_sequences=True))
    model.add(LayerNormalization())
    model.add(ConvLSTM2D(64, (3, 3), padding="same", return_sequences=True))
    model.add(LayerNormalization())

    # Decoder
    model.add(TimeDistributed(Conv2DTranspose(64, (3, 3), strides=2, padding='same')))
    model.add(LayerNormalization())
    model.add(TimeDistributed(Conv2DTranspose(128, (6, 6), strides=4, padding='same')))
    model.add(LayerNormalization())
    model.add(TimeDistributed(Conv2D(1, (5, 5), activation="sigmoid", padding='same')))
    model.compile(loss='mse', optimizer=Adam(learning_rate=1e-4))
    return model