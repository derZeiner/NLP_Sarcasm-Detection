import pandas as pd
import numpy as np
import json
import re
import string
import nltk
from nltk.corpus import stopwords
from num2words import num2words
import contractions
import matplotlib.pyplot as plt
import warnings
from sklearn.model_selection import train_test_split, KFold
import tensorflow as tf
from keras.preprocessing.text import Tokenizer
from keras.utils import pad_sequences
from keras.callbacks import EarlyStopping

# Ignore the warnings
warnings.filterwarnings('ignore')
# Global variables that will be used in the functions
# Download the stopwords
nltk.download('stopwords')
# Create a set of stopwords
stop_words = set(stopwords.words('english'))
# Create a list of punctuation
punctuation = list(string.punctuation)
# Create a set of stopwords and punctuation
stop_words.update(punctuation)


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
    # Drop the article_link column
    _data.drop(['article_link'], axis=1, inplace=True)
    # Drop the rows with missing values
    _data.dropna(inplace=True)
    # Drop the duplicate rows
    _data.drop_duplicates(subset=['headline'], inplace=True)
    # Clean the text in the headline column
    _data['headline'] = _data['headline'].apply(clean_row_text)
    # Create a new column that contains the length of each sentence
    _data['sentence_length'] = _data['headline'].str.len()
    # Return the cleaned data
    return _data


def clean_row_text(_text):
    # ***********************************
    # Function: Cleans a row of text
    # ***********************************
    # remove the square brackets from the text
    _text = re.sub('[[^]]*]', '', _text)
    # find all numbers in the text
    digits = re.findall(r'\d+', _text)
    # Convert the numbers to words
    for digit in digits:
        text = num2words(int(digit))
        _text = _text.replace(digit, text)
    # Convert the contractions to their expanded forms -> I'm = I am
    _text = contractions.fix(_text)
    # Remove the stop words
    final_text = []
    for i in _text.split():
        if i.strip().lower() not in stop_words:
            final_text.append(i.strip())

    return " ".join(final_text)


def data_info(_data):
    # ***********************************
    # Function: Prints the data info
    # ***********************************
    # Print the first 10 rows of the data
    print("Head:")
    print(_data.head())
    # Print the data types of the columns
    print("Info:")
    print(_data.info())
    # Print the number of missing values in each column
    print(f"NA-Data: {_data.isna().sum()}")
    # Print the number of rows and columns in the data
    print(f"Shape: {_data.shape}")
    # Print the number of duplicate rows in the data
    print(f"The number of duplicated rows is: {_data.duplicated().sum()}")
    # Print length of the longest sentence
    print(f"Length of the longest sentence: {_data['sentence_length'].max()}")


def plot_graphs(_history, _string, _name, _type):
    # ***********************************
    # Function: Plots the graphs
    # ***********************************
    # Plot the graphs
    plt.plot(_history.history[_string])
    plt.plot(_history.history['val_' + _string])
    plt.xlabel("Epochs")
    plt.ylabel(_string)
    plt.legend([_string, 'val_' + _string])
    plt.title("Sarcasm Detection - " + _name + " - " + _type)
    plt.savefig(_name + "_" + _type + ".png")
    plt.show()


def train_model_from_scratch(_data, num_folds, num_epochs, vocab_size, max_length, embedding_dim, padding_type, es):
    # ***********************************
    # Function: Trains the model from scratch using k-fold cross validation
    # ***********************************
    # clean the data
    _data = clean_data(_data)
    # Print the data info
    data_info(_data)
    # Split the data into train and test sets
    x_train, x_test, y_train, y_test = train_test_split(np.array(_data.headline), np.array(_data.is_sarcastic), test_size=0.2)

    # Create the tokenizer
    tokenizer = Tokenizer(num_words=vocab_size, oov_token="<OOV>")
    # Fit the tokenizer on the training data
    tokenizer.fit_on_texts(x_train)
    # Create the word index
    word_index = tokenizer.word_index
    # Create the training sequences
    train_sequences = tokenizer.texts_to_sequences(x_train)
    # Pad the training sequences
    train_padded = pad_sequences(train_sequences, maxlen=max_length, padding=padding_type, truncating=padding_type)
    # Create the test sequences
    test_sequences = tokenizer.texts_to_sequences(x_test)
    # Pad the test sequences
    test_padded = pad_sequences(test_sequences, maxlen=max_length, padding=padding_type, truncating=padding_type)

    # kfold cross validation
    # Create the kfold object
    kfold = KFold(n_splits=num_folds, shuffle=True)
    # Create the list that will contain the scores
    scores = []
    # Create the list that will contain the histories
    histories = []
    # Loop through the folds
    for train, test in kfold.split(train_padded, y_train):
        print("-" * 50)
        print("Training on fold " + str(len(scores) + 1) + "/" + str(num_folds) + "...")
        # Create the model
        model = tf.keras.Sequential([
            tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length=max_length),
            tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(128, return_sequences=True)),
            tf.keras.layers.GlobalMaxPool1D(),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(32, activation='sigmoid'),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])
        # Compile the model
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

        if es:
            # define early stopping callback to stop training when the validation loss starts to increase
            early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
            # Train the model
            history = model.fit(train_padded[train], y_train[train], epochs=num_epochs,
                                validation_data=(train_padded[test], y_train[test]), verbose=1,
                                callbacks=[early_stopping])
        else:
            # Train the model
            history = model.fit(train_padded[train], y_train[train], epochs=num_epochs,
                                validation_data=(train_padded[test], y_train[test]), verbose=1)

        # Evaluate the model
        scores.append(model.evaluate(train_padded[test], y_train[test], verbose=0))
        # Append the history
        histories.append(history)

    # Print the scores
    print(f"Scores: {scores}")
    # Print the mean score
    print(f"Mean score: {np.mean(scores)}")
    # Print the standard deviation of the scores
    print(f"Standard deviation of the scores: {np.std(scores)}")
    # Plot the graphs
    plot_graphs(histories[0], 'accuracy', 'KFold-Scratch-Model', 'Accuracy')
    plot_graphs(histories[0], 'loss', 'KFold-Scratch-Model', 'Loss')
    # Evaluate the model on the test data
    print(f"Test loss: {model.evaluate(test_padded, y_test)[0]}")
    print(f"Test accuracy: {model.evaluate(test_padded, y_test)[1]}")
    # Save the model
    model.save('../../models/scratch_sarcasm_kfold_model.h5')

def run():
    # path to the dataset
    filepath = '../dataset/Sarcasm_Headlines_Dataset_v2.json'
    # Read the dataset
    data = read_file(filepath)
    num_folds = 5
    num_epochs = 10
    vocab_size = 10000
    max_length = 500
    embedding_dim = 50
    padding_type = 'post'
    es = True
    # train the model from scratch with pre-trained word embeddings
    train_model_from_scratch(data, num_folds, num_epochs, vocab_size, max_length, embedding_dim, padding_type, es)

run()


