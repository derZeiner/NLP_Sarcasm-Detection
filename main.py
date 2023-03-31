import pandas as pd
import json
import re
import string
import nltk
from nltk.corpus import stopwords
from num2words import num2words
import contractions
import gensim
import keras
from keras.models import Sequential
from keras.utils import pad_sequences
from keras.preprocessing import text
from keras.layers import Dense, Embedding, LSTM, Bidirectional, GRU
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

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
    _text = re.sub('\[[^]]*\]', '', _text)
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


# Function to create weight matrix from word2vec gensim model
def get_weight_matrix(model, vocab):
    # total vocabulary size plus 0 for unknown words
    vocab_size = len(vocab) + 1
    # define weight matrix dimensions with all 0
    weight_matrix = np.zeros((vocab_size, 200))
    # step vocab, store vectors using the Tokenizer's integer mapping
    for word, i in vocab.items():
        if word in model.wv:
            weight_matrix[i] = model.wv[word]

    return weight_matrix


if __name__ == '__main__':
    # path to the dataset
    filepath = 'dataset/Sarcasm_Headlines_Dataset_v2.json'
    # Read the dataset
    data = read_file(filepath)
    # Clean the data
    data = clean_data(data)
    # Print the data info
    data_info(data)

    words = []
    for i in data.headline.values:
        words.append(i.split())

    # Dimension of vectors we are generating
    EMBEDDING_DIM = 200

    # Creating Word Vectors by Word2Vec Method (takes time...)
    w2v_model = gensim.models.Word2Vec(sentences=words, vector_size=EMBEDDING_DIM)

    tokenizer = text.Tokenizer(num_words=35000)
    tokenizer.fit_on_texts(words)
    tokenized_train = tokenizer.texts_to_sequences(words)
    x = pad_sequences(tokenized_train, maxlen=20)

    # Adding 1 because of reserved 0 index
    # Embedding Layer creates one more vector for "UNKNOWN" words, or padded words (0s). This Vector is filled with zeros.
    # Thus our vocab size inceeases by 1
    vocab_size = len(tokenizer.word_index) + 1

    # Getting embedding vectors from word2vec and usings it as weights of non-trainable keras embedding layer
    embedding_vectors = get_weight_matrix(w2v_model, tokenizer.word_index)

    # Defining Neural Network
    model = Sequential()
    # Non-trainable embeddidng layer
    model.add(Embedding(vocab_size, output_dim=EMBEDDING_DIM, weights=[embedding_vectors], input_length=20, trainable=True))
    # LSTM
    model.add(Bidirectional(LSTM(units=128, recurrent_dropout=0.3, dropout=0.3, return_sequences=True)))
    model.add(Bidirectional(GRU(units=32, recurrent_dropout=0.1, dropout=0.1)))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer=keras.optimizers.Adam(lr=0.01), loss='binary_crossentropy', metrics=['acc'])
    model.summary()

    x_train, x_test, y_train, y_test = train_test_split(x, data.is_sarcastic, test_size=0.3, random_state=0)
    history = model.fit(x_train, y_train, batch_size=128, validation_data=(x_test, y_test), epochs=10)

    print("Accuracy of the model on Training Data is - ", model.evaluate(x_train, y_train)[1] * 100)
    print("Accuracy of the model on Testing Data is - ", model.evaluate(x_test, y_test)[1] * 100)

    # generate predictions on your testing data
    y_pred = model.predict(x_test)
    # convert predictions from one-hot encoding to class labels
    y_pred = np.argmax(y_pred, axis=1)
    # calculate F1 score
    f1 = f1_score(y_test, y_pred, average='weighted')
    print("F1 score: {:.2f}".format(f1))

