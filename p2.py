"""
The file data.json contains (query, intent) pairs for a simple dialog system.
A particular intent contains numerous example queries. For instance, the "balance"
intent might have queries "what is my balance?", "how much money do I have?", etc.

An outlier is a query that does not fit within an intent. By "fit", we mean that
the query is not semantically valid for that intent class. An outler might belong
to a different intent class, or to none.

For example, for the "balance" intent, outlier queries might be
"what is my routing number" and "will Trump be re-elected?".

For this problem, write a program to predict the outliers for each intent class.
The program should produce a file called 'outliers.json', which should have this
structure:

{
"<intent>": ["<outlier>", "<outlier>", "<outlier>", ... ],
"<intent>": ["<outlier>", "<outlier>", "<outlier>", ...]
}

Note that some intents might not have any outliers. Note also that there is not
one perfect solution to this problem. This is somewhat of an open-ended task.

Your program should be able to run using the command :
python p2.py data.json
"""

import json
from os import getcwd
from sys import argv

from data_silo.data_processing import DataProcessing
from model_brewery.d2v_distillery import BrewModel


def aggregate_intent_outlier(data_df):
    """
    Group Query by Intent + Doc2Vec Score below score to indicate possible anomaly.
    Dump into json file as the final step.

    :param data_df:     ( DataFrame ) Data of Doc2Vec Intent not the same as Original Intent
    """

    # Init Outlier Dictionary
    outlier_dict = {}
    # Get the list of unique Intents
    unique_intent_list = list(set(data_df["Intent"]))
    # Group by Intent w/ Doc2Vec Score below threshold
    for intent_key in unique_intent_list:
        subset_intent_outlier = data_df[(data_df["Intent"] == intent_key) & (data_df["Doc2Vec_Score"] < 0.6)]
        # Get the list of possible outlier queries
        outlier_dict[intent_key] = list(subset_intent_outlier["Query"])
    # Dump result to JSON file
    with open("outlier.json", "w") as outlier_file:
        json.dump(outlier_dict, outlier_file, indent=4)


def main():
    # Input JSON data
    incoming_json_data = f"{getcwd()}/data_lake/{argv[1]}"
    # Pre-process texts
    DataProcessing(incoming_json_data).clean_text()
    # Build Doc2Vec Model
    BrewModel(incoming_json_data).brew_doc2vec()
    # Assign New Intents to original data with scores
    data_tag_DF = BrewModel(incoming_json_data).brew_tags()
    # Get the queries w/ a different tag being assigned by Doc2Vec
    outliers_DF = data_tag_DF[data_tag_DF["Intent"] != data_tag_DF["Doc2Vec_Intent"]].reset_index(drop=True)
    # Aggregate potential outlier queries with score less than 0.4
    aggregate_intent_outlier(outliers_DF)


if __name__== "__main__":
    main()