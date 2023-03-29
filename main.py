import pandas as pd
import json
import re
import string


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
    # Clean the text in the headline column
    _data['headline'] = _data['headline'].apply(clean_row_text)
    # Drop the duplicate rows
    _data.drop_duplicates(subset=['headline'], inplace=True)
    # Return the cleaned data
    return _data


def clean_row_text(_text):
    # ***********************************
    # Function: Cleans a row of text
    # ***********************************
    _text = re.sub(r'\d+', '', _text)
    _text = "".join([char for char in _text if char not in string.punctuation])
    return _text


def data_info(_data):
    # ***********************************
    # Function: Prints the data info
    # ***********************************
    # Print the first 10 rows of the data
    print(_data.head())
    # Print the data types of the columns
    print(_data.info())
    # Print the number of missing values in each column
    print(_data.isna().sum())
    # Print the number of rows and columns in the data
    print(_data.shape)
    # Print the number of duplicate rows in the data
    print(_data.duplicated().sum())


if __name__ == '__main__':
    # path to the dataset
    filepath = 'dataset/Sarcasm_Headlines_Dataset_v2.json'
    # Read the dataset
    data = read_file(filepath)
    # Clean the data
    data = clean_data(data)
    # Print the data info
    data_info(data)



