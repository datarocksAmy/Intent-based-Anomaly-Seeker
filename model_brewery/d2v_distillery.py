from os import getcwd

import pandas as pd
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from nltk.tokenize import word_tokenize
from pyarrow.feather import read_feather

from data_silo.data_processing import DataProcessing


class BrewModel():
    def __init__(self):
        self.clean_query_df = read_feather(f"{getcwd()}/clean_query_data.feather")
        self.max_epochs = 20
        self.doc2vec_alpha = 0.025
        self.doc2vec_min_alpha = 0.025
        self.learning_rate = 0.0002


    def brew_doc2vec(self):
        # Init tagged document list
        tag_doc_list = []
        # Tagged each Query w/ original Intent
        for row_idx in range(len(self.clean_query_df)):
            tag_doc_list.append(TaggedDocument(words=word_tokenize(self.clean_query_df["Query"][row_idx].lower()),
                                            tags=[self.clean_query_df["Intent"][row_idx]]))
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
        doc2vec_model.save(f"{getcwd()}/model_shelf/d2v_intent_clean.model")


    def brew_score(self, query_text):
        doc2vec_model = Doc2Vec.load(f"{getcwd()}/model_shelf/d2v_intent_clean.model")
        processed_sentence = (DataProcessing().normalize(query_text)).split(" ")
        most_similar_result = doc2vec_model.docvecs.most_similar(
            positive=[doc2vec_model.infer_vector(processed_sentence)], topn=1)
        new_intent, intent_similarity_score = most_similar_result[0]

        return [new_intent, intent_similarity_score]


    def brew_tags(self):
        self.clean_query_df["Doc2Vec_Intent_Score"] = self.clean_query_df["Query"].apply(self.brew_score)
        self.clean_query_df[["Doc2Vec_Intent", "Doc2Vec_Score"]] = pd.DataFrame(self.clean_query_df
                                                                                ["Doc2Vec_Intent_Score"].tolist(),
                                                                                index=self.clean_query_df.index)
        self.clean_query_df.drop("Doc2Vec_Intent_Score", axis=1, inplace=True)

        return self.clean_query_df