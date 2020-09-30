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

from model_brewery.d2v_distillery import BrewModel


def aggregate_intent_outlier(data_df):
    outlier_dict = {}
    unique_intent_list = list(set(data_df["Intent"]))
    for intent_key in unique_intent_list:
        subset_intent_outlier = data_df[data_df["Intent"] == intent_key]
        outlier_dict["intent_key"] = list(subset_intent_outlier["Query"])

    return outlier_dict


def main():
    # DataProcessing(argv[1]).clean_text()
    data_tag_DF = BrewModel("data_lake/data.json").brew_tags()
    data_tag_DF["IsSame"] = (data_tag_DF["Intent"] == data_tag_DF["Doc2Vec_Intent"])
    outliers_DF = data_tag_DF[data_tag_DF["IsSame"] == True].reset_index(drop=True)
    agg_outliers_dict = aggregate_intent_outlier(outliers_DF)
    with open("outlier.json", "w") as outlier_file:
        json.dump(agg_outliers_dict, outlier_file, indent=4)

if __name__== "__main__":
    main()