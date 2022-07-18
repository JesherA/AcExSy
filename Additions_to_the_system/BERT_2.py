"""
First script for testing BERT 27/04/2022
"""
from typing import Optional

from pydantic import validate_arguments, PositiveInt
from typing_extensions import Literal

from Logger import logging
from run_config import RunConfig
from inputters import TrainOutDataManager, InputArticle

from .._base import TextRepresentatorAcronymIndependent, TextRepresentatorFactory
from helper import ExecutionTimeObserver, TrainInstance

# Bert specific packages
from sentence_transformers import SentenceTransformer, models
import torch.nn as nn

# replace typing_extensions by typing in python 3.7+
logger = logging.getLogger(__name__)


class FactoryBERT(
    TextRepresentatorFactory
):  # pylint: disable=too-few-public-methods
    """
    Text representator factory to test different BERT models
    """

    @validate_arguments
    def __init__(  # pylint: disable=too-many-arguments
        self,
        run_config: Optional[RunConfig] = RunConfig(),
    ):

        self.run_config = run_config

    def get_text_representator(
        self,
        train_data_manager: TrainOutDataManager,
        execution_time_observer: Optional[ExecutionTimeObserver] = None,
    ):
        """
        Main function for Bert. Loads model and creates the embedding vectors
        :param train_data_manager:
        :param execution_time_observer:
        :return:
        """
        # BERT model loader
        # bert_model = SentenceTransformer('pdelobelle/robbert-v2-dutch-base')
        # bert_model = SentenceTransformer('google/bert_uncased_L-4_H-256_A-4')

        word_emb = models.Transformer('pdelobelle/robbert-v2-dutch-base')
        pooling_model = models.Pooling(word_emb.get_word_embedding_dimension(),
                                       pooling_mode_mean_tokens=True)

        bert_model = SentenceTransformer(modules=[word_emb, pooling_model])

        # create embeddings for raw articles
        train_docs_emb = bert_model.encode(list(train_data_manager.get_raw_articles_db().values()),
                                           batch_size=150,
                                           normalize_embeddings=False)

        # get the wiki article ID
        invert_idx_article_id = {article_id: idx for idx, article_id in enumerate(train_data_manager.get_raw_articles_db().keys())}

        return _RepresentatorBERT(bert_model, train_docs_emb, invert_idx_article_id)


class _RepresentatorBERT(TextRepresentatorAcronymIndependent):
    def __init__(self, bert_model, articles_db, embeddings):
        super().__init__()
        self.articles_db = articles_db
        self.bert_model = bert_model
        self.embeddings = embeddings

    def _transform_input_text(self, article: InputArticle):
        """
        creates the embeddings vector for the test documents
        :param article: Input article that contains text (either raw or preprocessed)
        :return: np ndarray with the embedding for BERT
        """
        return self.bert_model.encode(article.get_raw_text())

    def _transform_train_instance(self, train_instance: TrainInstance):
        """
        creates the embedding factor for the train documents based on their id's
        :param train_instance: all attributes of the training data (article_id, acronym, expansion)
        :return: the xx of a training document based on the article id
        """
        return self.articles_db[self.embeddings.get(train_instance.article_id)]

