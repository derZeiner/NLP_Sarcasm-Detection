import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
import warnings
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import tensorflow as tf
import transformers
import tqdm

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
    model.save('../../models/pretrained_bert_model.h5')


# path to the dataset
filepath = '../dataset/Sarcasm_Headlines_Dataset_v2.json'
# Read the dataset
data = read_file(filepath)
# train the model with pre-trained BERT model
train_pretrained_model(data)
