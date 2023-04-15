import pandas as pd
import json
import matplotlib.pyplot as plt
import warnings
from transformers import DistilBertTokenizerFast, TFDistilBertForSequenceClassification
from sklearn.model_selection import train_test_split
import tensorflow as tf

# Ignore the warnings
warnings.filterwarnings('ignore')


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


def plot_graphs(_history, _string):
    # ***********************************
    # Function: Plots the graphs
    # ***********************************
    # Plot the graphs
    plt.plot(_history.history[_string])
    plt.plot(_history.history['val_' + _string])
    plt.xlabel("Epochs")
    plt.ylabel(_string)
    plt.legend([_string, 'val_' + _string])
    plt.show()


def train_pretrained_model(_data):
    # ***********************************
    # Function: Trains the pretrained model
    # ***********************************
    # Load the tokenizer
    tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')
    # split the data into train and test
    X_train, X_test, y_train, y_test = train_test_split(_data['headline'].values.tolist(), data['is_sarcastic'].values.tolist(), test_size=0.2, random_state=42)
    # Tokenize the data
    X_train_encodings = tokenizer(X_train, truncation=True, padding=True, max_length=16)['input_ids']
    X_test_encodings = tokenizer(X_test, truncation=True, padding=True, max_length=16)['input_ids']
    # Load the model
    model = TFDistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=2)
    # optimizer
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-5)
    # Compile the model
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
    # Train the model
    history = model.fit(X_train_encodings, y_train, epochs=3, batch_size=32, validation_data=(X_test_encodings, y_test))
    # evaluate the model
    # Plot the accuracy and loss
    plot_graphs(history, "accuracy")
    plot_graphs(history, "loss")

    # Save the trained model to disk
    model.save('../../models/pretrained_bert_model_2.h5')


# path to the dataset
filepath = '../dataset/Sarcasm_Headlines_Dataset_v2.json'
# Read the dataset
data = read_file(filepath)
# train the model with pre-trained BERT model
train_pretrained_model(data)
