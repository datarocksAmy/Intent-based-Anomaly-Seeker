from os import getcwd

from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from nltk.tokenize import word_tokenize
from pyarrow.feather import read_feather


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
        doc2vec_model.save("d2v_intent_clean.model")