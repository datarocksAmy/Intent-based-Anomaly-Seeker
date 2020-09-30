import json
import string

import pandas as pd
from nltk import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer, WordNetLemmatizer
from pyarrow.feather import write_feather


class DataProcessing:
    def __init__(self, json_arg=None):
        self.json_arg = json_arg
        self.data_df = None


    def retrieve_process_json(self):
        """
        Input data from json file.
        Convert list of lists into DataFrame and assign corresponding column names.

        :return:    ( DataFrame ) Input Data DataFrame w/ Columns Query & Intent
        """

        # Read JSON File
        with open(self.json_arg) as j_file:
            # Load in data
            data_json = json.load(j_file)
            # Convert List of Lists into DataFrame + Assign Columns Names : Query & Intent
            self.data_df = pd.DataFrame(data_json, columns=["Query", "Intent"])

            return self.data_df


    def normalize(self, row):
        """
        Pre-processing query strings.
        Remove punctuations, sentence + word tokenization, remove stop words, lemmitization, stemming.

        :param row: ( DataFrame row-wise ) Query string
        :return:    ( String ) Cleaned query string
        """

        # Ensure all texts are lower case
        lower_case_text = row.lower()
        # Split into sentences
        sentence_tokenize_text = " ".join(sent_tokenize(lower_case_text))
        # Remove punctuation
        remove_punc_text = sentence_tokenize_text.translate(str.maketrans('', '', string.punctuation)).upper()
        # Tokenize words
        tokenize_text = word_tokenize(remove_punc_text)
        # Remove Stop words
        english_stopwords_corpus = stopwords.words("english")
        remove_stopwords_text = [word for word in tokenize_text if word not in english_stopwords_corpus]
        # Lemmatization
        word_lemmatizer = WordNetLemmatizer()
        lemmatized_text = [word_lemmatizer.lemmatize(word) for word in remove_stopwords_text]
        # Stemming
        snowball_stemmer = SnowballStemmer("english")
        stemmed_text = [snowball_stemmer.stem(word) for word in lemmatized_text]
        # Take out numbers or any string containing numbers
        non_numerical_row = [x for x in stemmed_text if not any(c.isdigit() for c in x)]
        # Transform list into string
        clean_text = " ".join(non_numerical_row)

        return clean_text


    def clean_text(self):
        """
        Centralized data pre-processing function:
        - Load data from json file
        - Clean/normalized query string data
        - Write cleaned data with labels into feather file

        """

        # Get input data from json file
        self.retrieve_process_json()
        # Normalized Query strings
        self.data_df["Query"] = self.data_df["Query"].apply(self.normalize)
        # Write cleaned query data with corresponding intent into feather file
        write_feather(self.data_df, "clean_query_data.feather")
