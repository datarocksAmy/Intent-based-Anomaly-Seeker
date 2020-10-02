from __future__ import unicode_literals, print_function

import io
import json
import string
from os import getcwd
from random import sample

import pandas as pd
import ruamel.yaml
from nltk import ngrams
from pyarrow.feather import write_feather
from snips_nlu import SnipsNLUEngine
from snips_nlu.default_configs import CONFIG_EN
from spacy import load

from data_silo.data_processing import DataProcessing


class BrewSnips:
    def __init__(self, json_arg):
        self.nlu_engine = None
        self.json_arg = json_arg

    def simple_text_cleaner(self, query_text):
        """
        Replace, encode, remove unnecessary texts to ensure alignment w/ yaml format.

        :param query_text:  ( String ) Query
        :return:            ( String ) Cleaned-format query
        """

        # Encode into "utf-8" format
        query_text = query_text.encode(encoding='utf-8').decode("utf-8")
        # Replace $, %, + into relative words
        query_text = query_text.replace("$", "")
        query_text = query_text.replace("%", "percent")
        query_text = query_text.replace("+", "plus")
        # Remove punctuations
        query_text = query_text.translate(str.maketrans('', '', string.punctuation))
        # Replace "’" to avoid parsing error
        query_text = query_text.replace("’", "")

        return query_text

    def parse_snips_intent(self):
        """
        Parse original data.json into Snips NLU Engine Training Data in yaml format.
        Convert into yaml file through command line prompt :
        'snips-nlu generate-dataset en input-yaml-file > output-json-file'
        """

        # Get original data.json in DataFrame
        data_df = DataProcessing(f"{getcwd()}/data_lake/{self.json_arg}").retrieve_process_json()
        # Get list of Unique Intents
        intent_list = list(set(data_df["Intent"]))
        # Load SpaCy NLP Large Corpus
        spacy_nlp_engine = load('en_core_web_lg')
        # Init yaml object
        yaml = ruamel.yaml.YAML()
        # Set explicit start to True
        yaml.explicit_start = True
        # Parse by Intents
        for intent_name in intent_list:
            # yes and no are reserved values for yaml file.
            # To avoid parsing error, "_" is added before the intent name.
            if intent_name == "yes" or intent_name == "no":
                intent_dict = {"type": "intent", "name": f"{intent_name}s"}
            else:
                intent_dict = {"type": "intent", "name": intent_name}
            # Init Lists for Slots + Utterances
            slots_value_list = []
            utt_value_list = []
            # Subset current Intent Data
            subset_data = data_df[data_df["Intent"] == intent_name].reset_index(drop=True)
            # Get current Intent Queries
            intent_query_words = list(subset_data["Query"])
            # Get the 4 grams and convert into a list
            word_ngrams = (pd.Series(ngrams(intent_query_words, 4))).to_list()
            # Random sample 80% of each Intent as training phrases for NLU Engine
            sample_ngrams = sample(word_ngrams, int(len(subset_data)*0.8))
            # Start parsing each queries
            for phrases in sample_ngrams:
                # Join phrases back to one single sentence
                full_text = " ".join(phrases)
                # Parse Entity of the text through Spacy NLP Engine
                parse_phrases = spacy_nlp_engine(full_text)
                # Set slots
                if len(parse_phrases.ents) > 0:
                    # Get Entity Label and Text, if any
                    for nlp_entity in parse_phrases.ents:
                        entity_label = nlp_entity.label_
                        entity_text = nlp_entity.text
                        # Construct "slot" for name and entity
                        slot_entities = {"name": entity_label, "entity": entity_label}
                        # Replace text with entity label
                        full_text = full_text.replace(entity_text, f"[{entity_label}]({entity_text})")
                        # Store "utterances" from the ngram
                        utt_value_list.append(full_text)
                        # Store unique "slots"
                        if slot_entities not in slots_value_list:
                            slots_value_list.append(slot_entities)
            # Set slots in intent dictionary
            if len(slots_value_list) > 0:
                intent_dict["slots"] = slots_value_list
            # Set utterances in intent dictionary
            if len(utt_value_list) > 0:
                intent_dict["utterances"] = utt_value_list
            # If there's no utterances found, use the original ngrams
            else:
                intent_dict["utterances"] = [" ".join(gram) for gram in sample_ngrams]
            # Append into output yaml
            with open(f"{getcwd()}/data_lake/intent_ngram.yaml", "a") as file:
                yaml.dump(intent_dict, file)

    def get_nlu_engine(self):
        """
        Get JSON file with our intents and entities tag from part of our input data.

        Update the Snips NLU Engine.

        :return:    ( Snips NLU Object ) Customized Snips NLU Engine
        """

        # Get parsed intent ngram JSON file
        with io.open(f"{getcwd()}/data_lake/intent_ngram.json") as f:
            custom_dataset = json.load(f)
        # Init Snips NLU Engine
        self.nlu_engine = SnipsNLUEngine(config=CONFIG_EN)
        # Fit sample data into NLU Engine for customization
        self.nlu_engine = self.nlu_engine.fit(custom_dataset)

        return self.nlu_engine

    def parse_intent_name_prob(self, text):
        """
        Parse New IntentName and Probability based on customized Snips NLU Engine.

        :param text:    ( String ) Query
        :return:        ( List ) Intent Name and Probability Score
        """

        # Parse queries to determine intent name and probability score
        parsing = self.nlu_engine.parse(text)
        # Get intent name
        intent_name = parsing["intent"]["intentName"]
        # get the probability
        intent_prob = parsing["intent"]["probability"]

        return [intent_name, intent_prob]

    def brew_intent_score(self):
        """
        Generate new intent name and probability score for each query.
        Parse them into JSON format and write into a JSON file.
        """

        # Get Customized Snips NLU Engine
        self.get_nlu_engine()
        # Get the original json data
        with io.open(f"{getcwd()}/data_lake/{self.json_arg}") as f:
            data_df = json.load(f)
            # Convert list of lists into DataFrame w/ relative columns
            data_content_df = pd.DataFrame(data_df, columns=["Query", "Intent"])
            # Set Intent Similarity Score w/ New Intent
            data_content_df["NLU_Intent_Score"] = data_content_df["Query"].apply(self.parse_intent_name_prob)
            # Split into individual columns : Intent and Score
            data_content_df[["NLU_Intent", "NLU_Score"]] = pd.DataFrame(data_content_df["NLU_Intent_Score"].tolist(),
                                                                        index=data_content_df.index)
            # Drop unused column
            data_content_df.drop("NLU_Intent_Score", axis=1, inplace=True)
            # Store all output into feather file
            write_feather(data_content_df, f"{getcwd()}/data_lake/SnipsNLUData.feather")
