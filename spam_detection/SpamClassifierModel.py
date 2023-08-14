import numpy as np
import tensorflow as tf

class SpamClassifierModel(tf.keras.Model):
    def __init__(self, vocab_sz, embed_sz, input_length, num_filters, 
                 kernel_sz, output_sz, run_mode, embedding_weights, **kwargs):
        
        super(SpamClassifierModel, self).__init__(**kwargs)

        """
        scratch: learn the weights during the training
        vectorizer: transfer learning. set the embedding weights from our external matrix E but set the trainable parameter to False
        finetuning (else): set the embedding weights from our external matrix E, as well as set the layer trainable
        """
        if run_mode == "scratch":
            self.embedding = tf.keras.layers.Embedding(
                vocab_sz, 
                embed_sz, 
                input_length=input_length, 
                trainable=True)
        elif run_mode == "vectorizer":
            self.embedding = tf.keras.layers.Embedding(
                vocab_sz, 
                embed_sz, 
                input_length=input_length, 
                weights=[embedding_weights],
                trainable=False)
        else:
            self.embedding = tf.keras.layers.Embedding(
                vocab_sz, 
                embed_sz, 
                input_length=input_length, 
                weights=[embedding_weights],
                trainable=True)

        self.conv = tf.keras.layers.Conv1D(filters=num_filters, kernel_size=kernel_sz, activation="relu")
        self.dropout = tf.keras.layers.SpatialDropout1D(0.2)
        self.pool = tf.keras.layers.GlobalMaxPool1D()
        self.dense = tf.keras.layers.Dense(output_sz, activation="softmax")

    def call(self, x):
        x = self.embedding(x)
        x = self.conv(x)
        x = self.dropout(x)
        x = self.pool(x)
        x = self.dense(x)
        return x
