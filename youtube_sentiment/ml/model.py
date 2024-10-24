import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Embedding, LSTM, Bidirectional
import os, sys
import matplotlib.pyplot as plt
from youtube_sentiment.logger import logging
from youtube_sentiment.exception import YoutubeException
from youtube_sentiment.constants import DATA_TRANSFORMATION_PAD_SEQUENCES_MAX_LEN

from transformers import TFBertForSequenceClassification


def train_model(vocab_size):

    try:
        # model = Sequential()
        # model.add(
        #     Embedding(
        #         input_dim=vocab_size,
        #         output_dim=10,
        #         input_length=DATA_TRANSFORMATION_PAD_SEQUENCES_MAX_LEN,
        #     )
        # )
        # model.add(Bidirectional(LSTM(units=56, return_sequences=True)))
        # model.add(Bidirectional(LSTM(units=28, return_sequences=True)))
        # model.add(Bidirectional(LSTM(units=8, return_sequences=False)))
        # model.add(Dense(units=2, activation="softmax"))
        # model.compile(
        #     loss="sparse_categorical_crossentropy",
        #     optimizer="adam",
        #     metrics=["accuracy"],
        # )
        # model.build(input_shape=(None, DATA_TRANSFORMATION_PAD_SEQUENCES_MAX_LEN))

        # Load BERT tokenizer and pre-trained BERT model
        model = TFBertForSequenceClassification.from_pretrained(
            "bert-base-uncased", num_labels=2
        )  # Binary classification

        model.compile(
            # optimizer=tf.keras.op.Adam(learning_rate=5e-5),
            optimizer="adam",
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=["accuracy"],
        )

        logging.info(model.summary())

        return model
    except Exception as e:
        raise YoutubeException(e, sys)


def plot_accuracy_and_loss_graph(history, accuracy_filename, loss_filename):
    try:
        logging.info("Creating the plots directory")
        dir_name = os.path.dirname(accuracy_filename)
        os.makedirs(dir_name, exist_ok=True)

        # Plot training & validation accuracy values
        plt.figure(figsize=(12, 6))

        # Plot accuracy
        # plt.subplot(1, 2, 1)
        plt.plot(history.history["accuracy"], label="Train Accuracy")
        plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
        plt.title("Model Accuracy")
        plt.ylabel("Accuracy")
        plt.xlabel("Epoch")
        plt.legend(loc="upper right")

        # Save accuracy plot
        plt.savefig(accuracy_filename)

        # Plot training & validation loss values
        plt.figure(figsize=(12, 6))
        plt.plot(history.history["loss"], label="Train Loss")
        plt.plot(history.history["val_loss"], label="Validation Loss")
        plt.title("Model Loss")
        plt.ylabel("Loss")
        plt.xlabel("Epoch")
        plt.legend(loc="upper right")

        # Save loss plot
        plt.savefig(loss_filename)

    except Exception as e:
        raise YoutubeException(e, sys)
