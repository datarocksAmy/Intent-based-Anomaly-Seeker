from __future__ import unicode_literals, print_function
import io
import json
from snips_nlu import SnipsNLUEngine
from snips_nlu.default_configs import CONFIG_EN
from os import getcwd
import pandas as pd
from pyarrow.feather import write_feather


class BrewSnips:
    def __init__(self):
        self.nlu_engine = None

    def get_nlu_engine(self):
        """

        :return:
        """
        with io.open(f"{getcwd()}/data_lake/intent_ngram.json") as f:
            sample_dataset = json.load(f)
        self.nlu_engine = SnipsNLUEngine(config=CONFIG_EN)
        self.nlu_engine = self.nlu_engine.fit(sample_dataset)
        return self.nlu_engine

    def parse_intent_name_prob(self, text):
        self.get_nlu_engine()
        parsing = self.nlu_engine.parse(text)
        intent_name = (parsing["intent"]["intentName"])
        intent_prob = (parsing["intent"]["probability"])
        return [intent_name, intent_prob]


    def brew_intent_score(self):
        with io.open(f"{getcwd()}/data_lake/data.json") as f:
            data_df = json.load(f)
            data_content_df = pd.DataFrame(data_df, columns=["Query", "Intent"])
            print(data_content_df.head())
            # Set Intent Similarity Score w/ New Intent
            data_content_df["NLU_Intent_Score"] = data_content_df["Query"].apply(self.parse_intent_name_prob)
            # Split into individual columns : Intent and Score
            data_content_df[["NLU_Intent", "NLU_Score"]] = pd.DataFrame(data_content_df
                                                                                 ["NLU_Score"].tolist(),
                                                                                 index=data_content_df.index)
            # Drop unused column
            data_content_df.drop("NLU_Intent_Score", axis=1, inplace=True)
            data_content_df.to_excel("NLUOutput.xlsx", index=False)