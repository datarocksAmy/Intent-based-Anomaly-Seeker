from os import getcwd

import pandas as pd
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from nltk.tokenize import word_tokenize
from pyarrow.feather import read_feather

from data_silo.data_processing import DataProcessing


class BrewModel(DataProcessing):
    def __init__(self, json_arg):
        DataProcessing.__init__(self, json_arg)
        self.clean_query_df = read_feather(f"{getcwd()}/data_lake/clean_query_data.feather")
        self.max_epochs = 50
        self.doc2vec_alpha = 0.025
        self.doc2vec_min_alpha = 0.025
        self.learning_rate = 0.0002


    def brew_doc2vec(self):
        """
        Tagged documents (in this case, queries) with relative Intent.
        Build a simple Doc2Vec Model.
        Save model for further use.
        """

        # Init tagged document list
        tag_doc_list = []
        # Random Subset
        subset_clean_query_df = self.clean_query_df.groupby("Intent").sample(frac=.7).reset_index(drop=True)
        # Tagged each Query w/ original Intent
        for row_idx in range(len(subset_clean_query_df)):
            tag_doc_list.append(TaggedDocument(words=word_tokenize(subset_clean_query_df["Query"][row_idx].lower()),
                                            tags=[subset_clean_query_df["Intent"][row_idx]]))
        # Init simple base Doc2Vec Model
        doc2vec_model = Doc2Vec(epochs=self.max_epochs, alpha=self.doc2vec_alpha, min_alpha=self.doc2vec_min_alpha)
        # Build Vocab w/ Tagged Query - Intent
        doc2vec_model.build_vocab(tag_doc_list)
        # Train Doc2Vec Model for max_epochs
        for num_epoch in range(self.max_epochs):
            print(f">>>> TRAINING EPOCH : {num_epoch}")
            # Train Doc2Vec
            doc2vec_model.train(tag_doc_list, total_examples=doc2vec_model.corpus_count, epochs=self.max_epochs)
            # Decrease the learning rate
            doc2vec_model.alpha -= self.learning_rate
            # Set it as no decay
            doc2vec_model.min_alpha = doc2vec_model.alpha
        # Save trained Doc2Vec Model for further usage
        doc2vec_model.save(f"{getcwd()}/model_shelf/d2v_intent_clean50.model")


    def brew_score(self, query_text):
        """
        Get top 1 similarity score for each query from Doc2Vec Model.

        :param query_text:  ( String ) Processed query
        :return:            ( List ) New Intent from Doc2Vec Model with Similarity Score
        """

        # Obtain Trained Doc2Vec Model
        doc2vec_model = Doc2Vec.load(f"{getcwd()}/model_shelf/d2v_intent_clean50.model")
        # Process query to clean form and split sentence into words
        processed_sentence = (self.normalize(query_text)).split(" ")
        # Get the top 1 match intent with similarity score through Doc2Vec Model
        most_similar_result = doc2vec_model.docvecs.most_similar(
            positive=[doc2vec_model.infer_vector(processed_sentence)], topn=1)
        # Top 1 Intent w/ Similarity Score
        new_intent, intent_similarity_score = most_similar_result[0]

        return [new_intent, intent_similarity_score]


    def brew_tags(self):
        """
        Set and parse Doc2Vec Top 1 match Intent and Similarity Score for each query.

        :return:    ( DataFrame ) Original JSON Data in DataFrame format w/ New Intent + Similarity Score
                                    from Top1 Match Doc2Vec
        """

        # Get the original JSON data in DataFrame format
        original_json_df = self.retrieve_process_json()
        # Set Intent Similarity Score w/ New Intent
        original_json_df["Doc2Vec_Intent_Score"] = original_json_df["Query"].apply(self.brew_score)
        # Split into individual columns : Intent and Score
        original_json_df[["Doc2Vec_Intent", "Doc2Vec_Score"]] = pd.DataFrame(original_json_df
                                                                                ["Doc2Vec_Intent_Score"].tolist(),
                                                                                index=original_json_df.index)
        # Drop unused column
        original_json_df.drop("Doc2Vec_Intent_Score", axis=1, inplace=True)

        return original_json_df