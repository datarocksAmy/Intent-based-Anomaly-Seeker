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

from pyarrow.feather import read_feather

from model_brewery.snips_distilliery import BrewSnips


def aggregate_intent_outlier():
    """
    Group Query by Intent + Intent Score below score to indicate possible anomaly.
    Dump into json file as the final step.
    """

    # Init Outlier Dictionary
    outlier_dict = {}
    # Get Intent Tags + relative probabilities from feather
    snips_intent_data = read_feather(f"{getcwd()}/data_lake/SnipsNLUData.feather")
    snips_intent_diff_tag = snips_intent_data[(snips_intent_data["Intent"] != snips_intent_data["NLU_Intent"])]
    # Get the list of unique Intents
    unique_intent_list = list(set(snips_intent_diff_tag["Intent"]))
    # Group by Intent w/ NLU Score above threshold
    for intent_key in unique_intent_list:
        subset_intent_outlier = snips_intent_data[(snips_intent_data["Intent"] == intent_key) &
                                                  (snips_intent_data["NLU_Score"] > 0.95)]
        if not subset_intent_outlier.empty:
            # Get the list of possible outlier queries
            outlier_dict[intent_key] = list(subset_intent_outlier["Query"])
    # Dump result to JSON file
    with open(f"{getcwd()}/data_lake/outlier.json", "w") as outlier_file:
        json.dump(outlier_dict, outlier_file, indent=4)


def main():
    # Pass in input data to generate NLU intent and entities
    BrewSnips(argv[1]).brew_intent_score()
    # Generate outlier json file by intent
    aggregate_intent_outlier()


if __name__ == "__main__":
    main()