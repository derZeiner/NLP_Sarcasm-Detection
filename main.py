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
import tensorflow as tf
from keras.preprocessing.text import Tokenizer
from keras.utils import pad_sequences


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


def test_1(_data):
    # Clean the data
    _data = clean_data(_data)
    # Print the data info
    data_info(_data)

    # Split the data into train and test sets
    labels = np.array(_data.is_sarcastic)
    sentences = np.array(_data.headline)
    x_train, x_test, y_train, y_test = train_test_split(sentences, labels, test_size=0.2)

    # hyper-parameters
    vocab_size = 10000
    max_length = 32
    embedding_dim = 32
    padding_type = 'post'
    oov_token = '<OOV>'

    # tokenizing the texts
    tokenizer = Tokenizer(num_words=vocab_size, oov_token=oov_token)
    tokenizer.fit_on_texts(x_train)
    # padding the sequences
    train_sequences = tokenizer.texts_to_sequences(x_train)
    padded_train_sequences = pad_sequences(train_sequences, maxlen=max_length, padding=padding_type)
    test_sequences = tokenizer.texts_to_sequences(x_test)
    padded_test_sentences = pad_sequences(test_sequences, maxlen=max_length, padding=padding_type)

    # hyper-parameters
    number_of_epochs = 10
    filters = 128
    kernel_size = 5
    lr = 0.0001

    # model
    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length=max_length),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Conv1D(filters, kernel_size, activation='relu'),
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
    model.compile(loss='binary_crossentropy', optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
                  metrics=['accuracy'])
    # train the model
    history = model.fit(padded_train_sequences, y_train, epochs=number_of_epochs,
                        validation_data=(padded_test_sentences, y_test), verbose=1)
    # evaluate the model
    print('Accuracy on test set: ', model.evaluate(padded_test_sentences, y_test)[1] * 100)
    # Plot the accuracy and loss
    plot_graphs(history, "accuracy")
    plot_graphs(history, "loss")


def test_2(_data):
    from transformers import BertTokenizer, TFBertModel

    # Split the data into train and test sets
    labels = np.array(_data.is_sarcastic)
    sentences = np.array(_data.headline)
    x_train, x_test, y_train, y_test = train_test_split(sentences, labels, test_size=0.2)

    # Preprocess the data
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    train_encoded_data = tokenizer.batch_encode_plus(
        x_train,
        add_special_tokens=True,
        return_attention_mask=True,
        pad_to_max_length=True,
        max_length=256,
        truncation=True,
        return_tensors='tf'
    )

    test_encoded_data = tokenizer.batch_encode_plus(
        x_test,
        add_special_tokens=True,
        return_attention_mask=True,
        pad_to_max_length=True,
        max_length=256,
        truncation=True,
        return_tensors='tf'
    )

    # Create the BERT model
    bert = TFBertModel.from_pretrained('bert-base-uncased', trainable=False)

    input_ids = tf.keras.layers.Input(shape=(256,), dtype=tf.int32)
    attention_masks = tf.keras.layers.Input(shape=(256,), dtype=tf.int32)

    output = bert(input_ids, attention_mask=attention_masks)

    output = output.last_hidden_state[:, 0, :]

    output = tf.keras.layers.Dropout(0.3)(output)

    output = tf.keras.layers.Dense(1, activation='sigmoid')(output)

    model = tf.keras.models.Model(inputs=[input_ids, attention_masks], outputs=output)

    model.summary()

    # Train the model
    optimizer = tf.keras.optimizers.Adam(learning_rate=2e-5, epsilon=1e-08)

    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

    history = model.fit(
        x=[train_encoded_data['input_ids'], train_encoded_data['attention_mask']],
        y=y_train,
        validation_split=0.2,
        batch_size=32,
        epochs=3
    )

    # Evaluate the model on the test set
    _, test_acc = model.evaluate(
        x=[test_encoded_data['input_ids'], test_encoded_data['attention_mask']],
        y=y_test
    )

    print('Test accuracy:', test_acc)
    # Plot the accuracy and loss
    plot_graphs(history, "BERT-Model - accuracy")
    plot_graphs(history, "BERT-Model - loss")


if __name__ == '__main__':
    # path to the dataset
    filepath = 'dataset/Sarcasm_Headlines_Dataset_v2.json'
    # Read the dataset
    data = read_file(filepath)
    # Test 1
    test_1(data)
    # Test 2
    test_2(data)
