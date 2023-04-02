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
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import tensorflow as tf
from keras.preprocessing.text import Tokenizer
from keras.utils import pad_sequences
from keras.callbacks import EarlyStopping
import transformers
import tqdm

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
    # Drop the is_sarcastic column
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


def train_model_from_scratch(_data):
    # ***********************************
    # Function: Trains the model from scratch
    # ***********************************
    # clean the data
    _data = clean_data(data)
    # Print the data info
    data_info(_data)
    # Split the data into train and test sets
    x_train, x_test, y_train, y_test = train_test_split(np.array(_data.headline), np.array(_data.is_sarcastic), test_size=0.2)

    # hyper-parameters
    vocab_size = 10000
    max_length = 200
    embedding_dim = 64
    padding_type = 'post'

    # create a tokenizer
    tokenizer = Tokenizer(num_words=vocab_size, oov_token='<OOV>')
    # fit the tokenizer on the text
    tokenizer.fit_on_texts(x_train)
    # convert the train text to sequences
    train_sequences = tokenizer.texts_to_sequences(x_train)
    # pad the sequences
    padded_train_sequences = pad_sequences(train_sequences, maxlen=max_length, padding=padding_type)
    # convert the test text to sequences
    test_sequences = tokenizer.texts_to_sequences(x_test)
    # pad the sequences
    padded_test_sentences = pad_sequences(test_sequences, maxlen=max_length, padding=padding_type)
    # create the model
    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length=max_length),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Conv1D(128, 5, activation='relu'),
        tf.keras.layers.GlobalAveragePooling1D(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dropout(0.25),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(1, activation='sigmoid'),
    ])
    # model summary
    model.summary()
    # compile the model
    model.compile(loss='binary_crossentropy', optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001), metrics=['accuracy'])
    # define the early stopping callback
    es = EarlyStopping(monitor='val_loss', patience=3)
    # train the model
    history = model.fit(padded_train_sequences, y_train, epochs=10, validation_data=(padded_test_sentences, y_test), verbose=1, callbacks=[es])

    # evaluate the model
    # print the accuracy on the test set
    print('Accuracy on test set: ', model.evaluate(padded_test_sentences, y_test)[1] * 100)
    # Plot the accuracy and loss
    plot_graphs(history, "accuracy")
    plot_graphs(history, "loss")

    # Save the trained model
    model.save('trained_model_from_scratch.h5')


def create_bert_input_features(tokenizer, docs, max_seq_length):
    # ***********************************
    # Function: Creates the BERT input features
    # ***********************************
    # Tokenize all of the sentences and map the tokens to thier word IDs.
    all_ids, all_masks = [], []
    # For every sentence...
    for doc in tqdm.tqdm(docs, desc="Converting docs to features"):
        # `encode` will:
        #   (1) Tokenize the sentence.
        tokens = tokenizer.tokenize(doc)
        if len(tokens) > max_seq_length-2:
            tokens = tokens[0: (max_seq_length-2)]
        #  (2) Prepend the `[CLS]` token to the start.
        tokens = ['[CLS]'] + tokens + ['[SEP]']
        ids = tokenizer.convert_tokens_to_ids(tokens)
        masks = [1] * len(ids)
        # Zero-pad up to the sequence length.
        while len(ids) < max_seq_length:
            ids.append(0)
            masks.append(0)
        all_ids.append(ids)
        all_masks.append(masks)
    # Add the encoded sentence to the list.
    encoded = np.array([all_ids, all_masks])
    return encoded


def train_pretrained_model(_data):
    # ***********************************
    # Function: Trains the pretrained model
    # ***********************************
    # load the transformers tokenizer
    tokenizer = transformers.DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

    # create the bert input features
    inp_id = tf.keras.layers.Input(shape=(20,), dtype='int32', name="bert_input_ids")
    inp_mask = tf.keras.layers.Input(shape=(20,), dtype='int32', name="bert_input_masks")
    inputs = [inp_id, inp_mask]

    # load the pretrained model
    hidden_state = transformers.TFDistilBertModel.from_pretrained('distilbert-base-uncased')(inputs)[0]
    # get the pooled output
    pooled_output = hidden_state[:, 0]
    # create the model layers
    dense1 = tf.keras.layers.Dense(256, activation='relu')(pooled_output)
    drop1 = tf.keras.layers.Dropout(0.25)(dense1)
    dense2 = tf.keras.layers.Dense(256, activation='relu')(drop1)
    drop2 = tf.keras.layers.Dropout(0.25)(dense2)
    output = tf.keras.layers.Dense(1, activation='sigmoid')(drop2)

    # create the model
    model = tf.keras.Model(inputs=inputs, outputs=output)
    # compile the model
    model.compile(optimizer=tf.optimizers.Adam(learning_rate=2e-5, epsilon=1e-08), loss='binary_crossentropy', metrics=['accuracy'])
    # model summary
    model.summary()

    # Split the data into train and test sets
    x_train, x_test, y_train, y_test = train_test_split(np.array(_data.headline), np.array(_data.is_sarcastic), test_size=0.2, random_state=42)

    # create the bert input features
    train_features_ids, train_features_masks = create_bert_input_features(tokenizer, x_train, max_seq_length=20)
    # define the early stopping callback
    es = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3)
    # train the model
    history = model.fit([train_features_ids, train_features_masks], y_train, validation_split=0.2, epochs=5, batch_size=25, callbacks=[es], shuffle=True, verbose=1)

    # evaluate the model
    # print the accuracy on the test set
    test_features_ids, test_features_masks = create_bert_input_features(tokenizer, x_test, max_seq_length=20)
    predictions = [1 if pr > 0.5 else 0 for pr in model.predict([test_features_ids, test_features_masks], verbose=0).ravel()]
    print(classification_report(y_test, predictions))
    # Plot the accuracy and loss
    plot_graphs(history, "accuracy")
    plot_graphs(history, "loss")

    # Save the trained model to disk
    model.save('pretrained_bert_model.h5')


if __name__ == '__main__':
    # path to the dataset
    filepath = 'dataset/Sarcasm_Headlines_Dataset_v2.json'
    # Read the dataset
    data = read_file(filepath)
    # train the model from scratch with pre-trained word embeddings
    train_model_from_scratch(data)
    # train the model with pre-trained BERT model
    train_pretrained_model(data)
