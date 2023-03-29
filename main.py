import pandas as pd
import json
import numpy as np

from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Bidirectional, LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split


def read_file(path):
    # ***********************************
    # Function: Reads a json file and converts it to a pandas dataframe
    # Load the dataset
    # ***********************************
    # Read the json file
    with open(path, 'r') as f:
        # Load the data into a variable
        raw_data = [json.loads(line) for line in f]
    # Convert the data to a pandas dataframe and return it
    return pd.DataFrame(raw_data)


def clean_data(_data):
    # ***********************************
    # Function: Cleans the data
    # ***********************************
    # Drop the is_sarcastic column
    _data.drop(['article_link'], axis=1, inplace=True)
    # Return the cleaned data
    return _data


if __name__ == '__main__':
    # path to the dataset
    filepath = 'dataset/Sarcasm_Headlines_Dataset_v2.json'
    # Read the dataset
    data = read_file(filepath)
    # Clean the data
    data = clean_data(data)
    # Print the first 10 rows of the data
    print(data.head(10))

    # todo: one-hot encode the labels probieren
    # todo: Bag of Words probieren
    # todo: N-Grams probieren
    # todo: TF-IDF probieren


    # Preprocess the data
    max_words = 10000
    tokenizer = Tokenizer(num_words=max_words, oov_token='<OOV>')
    tokenizer.fit_on_texts(data['headline'])
    sequences = tokenizer.texts_to_sequences(data['headline'])

    # Pad the sequences
    max_length = 128
    padded_sequences = pad_sequences(sequences, maxlen=max_length, padding='post', truncating='post')

    # Prepare the input features and labels
    input_features = np.array(padded_sequences)
    labels = data['is_sarcastic'].values

    # Split the dataset into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(input_features, labels, test_size=0.2, random_state=42)

    # Create the model
    embedding_dim = 128

    model = Sequential([
        Embedding(max_words, embedding_dim, input_length=max_length),
        Bidirectional(LSTM(64, return_sequences=True)),
        Bidirectional(LSTM(32)),
        Dense(64, activation='relu'),
        Dense(1, activation='sigmoid')
    ])

    model.summary()

    # Train the model
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    early_stop = EarlyStopping(monitor='val_loss', patience=3)

    history = model.fit(
        X_train, y_train,
        batch_size=32,
        epochs=10,
        validation_data=(X_val, y_val),
        callbacks=[early_stop]
    )

    # Evaluate the model
    evaluation = model.evaluate(X_val, y_val)
    print(f"Validation Loss: {evaluation[0]}, Validation Accuracy: {evaluation[1]}")
