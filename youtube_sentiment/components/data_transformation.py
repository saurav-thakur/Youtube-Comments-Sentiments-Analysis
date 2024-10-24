import os, sys
import pandas as pd
import numpy as np

from youtube_sentiment.logger import logging
from youtube_sentiment.exception import YoutubeException
from youtube_sentiment.entity.config_entity import (
    DataTransformationConfig,
    DataIngestionConfig,
)
from youtube_sentiment.entity.artifact_entity import (
    DataValidationArtifact,
    DataTransformationArtifact,
    DataIngestionArtifact,
)
from youtube_sentiment.utils.utilities import (
    read_csv_data,
    read_yaml_file,
    save_preprocessed_object,
)
from youtube_sentiment.constants import (
    DATA_TRANSFORMATION_PAD_SEQUENCES_PADDING,
    DATA_TRANSFORMATION_PAD_SEQUENCES_MAX_LEN,
    DATA_TRANSFORMATION_POSITIVE_SENTIMENT_MAP,
    DATA_TRANSFORMATION_NEGATIVE_SENTIMENT_MAP,
)

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.utils import pad_sequences

from transformers import BertTokenizer

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")


class DataTransformation:
    def __init__(
        self,
        data_transformation_config: DataTransformationConfig,
        data_validation_artifact: DataValidationArtifact,
        data_ingestion_artifact: DataIngestionArtifact,
    ):
        try:
            self.data_transformation_config = data_transformation_config
            self.data_validation_artifact = data_validation_artifact
            self.data_ingestion_artifact = data_ingestion_artifact
            self.schema_file = read_yaml_file(
                self.data_transformation_config.schema_file
            )
        except Exception as e:
            raise YoutubeException(e, sys)

    @staticmethod
    def map_sentiments(df: pd.DataFrame) -> pd.DataFrame:
        """
        This function maps the sentiment values to numeric labels.

        Args:
            df (pd.DataFrame): The dataframe containing the data.
            target_column (str): The name of the target column with sentiment values.

        Returns:
            pd.DataFrame: DataFrame with the target column mapped to numeric labels.
        """
        sentiment_mapping = {
            "positive": DATA_TRANSFORMATION_POSITIVE_SENTIMENT_MAP,
            "negative": DATA_TRANSFORMATION_NEGATIVE_SENTIMENT_MAP,
        }
        df["label"] = df["label"].replace(sentiment_mapping)
        return df

    # Tokenize text
    def tokenize_function(data):
        return tokenizer(data["text"], padding="max_length", truncation=True)

    def initiate_transform_data(self) -> DataTransformationArtifact:
        try:
            if self.data_validation_artifact.validation_status:
                logging.info("loading the ingested data")

                train_data = read_csv_data(self.data_ingestion_artifact.train_file_path)
                test_data = read_csv_data(self.data_ingestion_artifact.test_file_path)

                logging.info("mapping sentiments")
                train_data = self.map_sentiments(train_data)
                test_data = self.map_sentiments(test_data)

                logging.info("sentiments mapped")

                logging.info(
                    "Concatenating train and test to make a single data for tokenization."
                )
                df = pd.concat([train_data, test_data], axis=0)

                logging.info("tokenizing the data")

                # tokenizer = Tokenizer()
                # tokenizer.fit_on_texts(df["text"])
                # logging.info(f"number of sentences or rows {tokenizer.document_count}")

                # train_data_tokenized = tokenizer.texts_to_sequences(train_data["text"])
                # test_data_tokenized = tokenizer.texts_to_sequences(test_data["text"])

                train_data_tokenized = train_data["text"].apply(
                    lambda x: self.tokenize_function(x)
                )
                test_data_tokenized = test_data["text"].apply(
                    lambda x: self.tokenize_function(x)
                )

                # Convert datasets to TensorFlow format
                train_dataset = train_data_tokenized.to_tf_dataset(
                    columns=["input_ids", "attention_mask"],
                    label_cols="label",
                    batch_size=8,
                    shuffle=True,
                )

                test_dataset = test_data_tokenized.to_tf_dataset(
                    columns=["input_ids", "attention_mask"],
                    label_cols="label",
                    batch_size=8,
                )

                # logging.info("saving the tokenizer")
                # dir_name = os.path.dirname(
                #     self.data_transformation_config.data_transformation_preprocessed_tokenizer
                # )
                # os.makedirs(dir_name, exist_ok=True)
                # save_preprocessed_object(
                #     self.data_transformation_config.data_transformation_preprocessed_tokenizer,
                #     tokenizer,
                # )
                # logging.info("tokenizer saved")

                # logging.info("padding the sequences")
                # train_data_padded_sequences = pad_sequences(
                #     sequences=train_data_tokenized,
                #     padding=DATA_TRANSFORMATION_PAD_SEQUENCES_PADDING,
                #     maxlen=DATA_TRANSFORMATION_PAD_SEQUENCES_MAX_LEN,
                # )
                # test_data_padded_sequences = pad_sequences(
                #     sequences=test_data_tokenized,
                #     padding=DATA_TRANSFORMATION_PAD_SEQUENCES_PADDING,
                #     maxlen=DATA_TRANSFORMATION_PAD_SEQUENCES_MAX_LEN,
                # )
                # logging.info(
                #     f"the shape of padded data is train: {train_data_padded_sequences.shape} and test: {test_data_padded_sequences.shape}"
                # )

                dir_name = os.path.dirname(
                    self.data_transformation_config.data_transformation_transformed_train_data
                )
                os.makedirs(dir_name, exist_ok=True)

                logging.info("saving the data as npy format")
                # np.save(
                #     self.data_transformation_config.data_transformation_transformed_train_data,
                #     train_data_padded_sequences,
                # )
                # np.save(
                #     self.data_transformation_config.data_transformation_transformed_test_data,
                #     test_data_padded_sequences,
                # )
                np.save(
                    self.data_transformation_config.data_transformation_transformed_train_data,
                    train_data_tokenized,
                )
                np.save(
                    self.data_transformation_config.data_transformation_transformed_test_data,
                    test_data_tokenized,
                )

                logging.info("saving the labels as npy format")
                np.save(
                    self.data_transformation_config.data_transformation_transformed_train_label,
                    train_data[self.schema_file["target_column"]].values,
                )
                np.save(
                    self.data_transformation_config.data_transformation_transformed_test_label,
                    test_data[self.schema_file["target_column"]].values,
                )

                logging.info("data and labels saved as numpy file")

                data_transformation_artifact = DataTransformationArtifact(
                    data_transformation_transformed_train_data=self.data_transformation_config.data_transformation_transformed_train_data,
                    data_transformation_transformed_test_data=self.data_transformation_config.data_transformation_transformed_test_data,
                    data_transformation_transformed_train_label=self.data_transformation_config.data_transformation_transformed_train_label,
                    data_transformation_transformed_test_label=self.data_transformation_config.data_transformation_transformed_test_label,
                    data_transformation_tokenizer=self.data_transformation_config.data_transformation_preprocessed_tokenizer,
                )
                return data_transformation_artifact
            else:
                raise Exception(self.data_validation_artifact.message)
        except Exception as e:
            raise YoutubeException(e, sys)
