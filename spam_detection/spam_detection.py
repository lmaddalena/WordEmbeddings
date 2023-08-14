import argparse
import gensim.downloader as api
import numpy as np
import os
import shutil
import tensorflow as tf
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix
from SpamClassifierModel import *

DATASET_URL = "https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip"
NUM_CLASSES = 2
EMBEDDING_DIM = 300
DATA_DIR = "data"
EMBEDDING_NUMPY_FILE = os.path.join(DATA_DIR, "E.npy")
EMBEDDING_MODEL = 'glove-wiki-gigaword-300'
NUM_EPOCHS = 3

# data distribution is 4827 ham and 747 spam (total 5574), which
# works out to approx 87% ham and 13% spam, so we take reciprocals
# and this works out to being each spam (1) item as being
# approximately 8 times as important as each ham (0) message
CLASS_WEIGHTS = {0: 1, 1: 8}

def download(url):

    local_file = os.path.join("datasets", "SMSSpamCollection")

    if not os.path.exists(local_file):
        local_file = url.split('/')[-1]
        p = tf.keras.utils.get_file(local_file, url, extract=True, cache_dir=".")

def read_file():
    labels, texts = [], []
    local_file = os.path.join("datasets", "SMSSpamCollection")

    with open(local_file, "r") as fin:
        for line in fin:
            label, text = line.strip().split('\t')
            labels.append(1 if label == "spam" else 0) 
            texts.append(text)
    
    return texts, labels

def build_embedding_matrix(word2idx, embedding_size, embedding_file):
    if os.path.exists(embedding_file):
        E = np.load(embedding_file)
    else:
        vocab_size = len(word2idx)
        E = np.zeros((vocab_size, embedding_size))
        word_vectors = api.load(EMBEDDING_MODEL)
        for word, idx in word2idx.items():
            try:
                E[idx] = word_vectors.word_vec(word)
            except KeyError:    # word not in embedding
                pass
        np.save(embedding_file, E)
    return E

def main():

    tf.random.set_seed(42)

    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", help="run mode", required=True,
        choices=[
            "scratch",
            "vectorizer",
            "finetuning"
        ])
    args = parser.parse_args()
    run_mode = args.mode

    # download the dataset
    download(DATASET_URL)

    # read the dataset into texts and labels
    texts, labels = read_file()
    print(f"texts contains {len(texts)} records")
    print(f"labels contains {len(labels)} records")

    # print data samples
    print(pd.DataFrame(texts, columns={"text"}).head())
    print(pd.DataFrame(labels, columns={"spam"}).head())

    # tokenize a list of texts into a list of sequence (list of integer)
    tokenizer = tf.keras.preprocessing.text.Tokenizer()
    tokenizer.fit_on_texts(texts)
    text_sequence = tokenizer.texts_to_sequences(texts)

    # pad sequence
    text_sequence = tf.keras.preprocessing.sequence.pad_sequences(text_sequence)

    num_records = len(text_sequence)
    max_seqlen = len(text_sequence[0])
    print("{:d} sentences, max length: {:d}".format(num_records, max_seqlen))

    # convert labels to categorical
    cat_labels = tf.keras.utils.to_categorical(labels, NUM_CLASSES)
    print("{:d} categorical labels".format(len(cat_labels)))
    print(pd.DataFrame(cat_labels, columns=["ham", "spam"]).head())

    # vocabulary
    word2idx = tokenizer.word_index
    idx2word = {v:k for k, v in word2idx.items()}
    word2idx["PAD"] = 0
    idx2word[0] = "PAD"
    vocab_size = len(word2idx)
    print("vocab size: {:d}".format(vocab_size))

    # print vocab head
    for i in range(10):
        print("[{:d}] {:s} ".format(i, idx2word[i]))
    print("...")

    # create the dataset for the network from padded sequence of integer
    # and categorical labels
    dataset = tf.data.Dataset.from_tensor_slices((text_sequence, cat_labels))

    # shuffle the dataset
    dataset = dataset.shuffle(10000)

    # split into test, val and train
    test_size = num_records // 4
    val_size = (num_records - test_size) // 10
    test_dataset = dataset.take(test_size)
    val_dataset = dataset.skip(test_size).take(val_size)
    train_dataset = dataset.skip(test_size + val_size)

    # set the batch size for each dataset
    BATCH_SIZE = 128
    test_dataset = test_dataset.batch(BATCH_SIZE, drop_remainder=True)
    val_dataset = val_dataset.batch(BATCH_SIZE, drop_remainder=True)
    train_dataset = train_dataset.batch(BATCH_SIZE, drop_remainder=True)

    """
    Gensim provides access to various trained embedding models.
    To get info run the following command from Python prompt: 
    >>> import gensim.downloader as api
    >>> api.info()

    to get models list:
    >>> api.info()['models'].keys()

        dict_keys([
        'fasttext-wiki-news-subwords-300', 
        'conceptnet-numberbatch-17-06-300', 
        'word2vec-ruscorpora-300', 
        'word2vec-google-news-300', 
        'glove-wiki-gigaword-50', 
        'glove-wiki-gigaword-100', 
        'glove-wiki-gigaword-200', 
        'glove-wiki-gigaword-300', 
        'glove-twitter-25', 
        'glove-twitter-50', 
        'glove-twitter-100', 
        'glove-twitter-200', 
        '__testing_word2vec-matrix-synopsis'
        ])

    We chose the 300d GloVe embeddings trained on Gigaword corpus
    in order to keep our model size small, we want to only consider embedding for words
    that exists on our vocabulary

    """
    E = build_embedding_matrix(word2idx, EMBEDDING_DIM, EMBEDDING_NUMPY_FILE)
    print("Embedding matrix: ", E.shape)
    
    # model definition
    conv_num_filters = 256
    conv_kernel_size = 3
    model = SpamClassifierModel(vocab_size, EMBEDDING_DIM, max_seqlen, conv_num_filters, conv_kernel_size, NUM_CLASSES, run_mode, E)
    model.build(input_shape=(None, max_seqlen))

    print(model.summary())

    # compile the model using the categorical cross entropy loss function and
    # the Adam optimizer
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

    # train the model
    model.fit(train_dataset, epochs=NUM_EPOCHS, validation_data=val_dataset, class_weight=CLASS_WEIGHTS)

    # evaluate against test set
    labels, predictions = [], []
    for Xtest, Ytest in test_dataset:
        Ytest_ = model.predict_on_batch(Xtest)
        ytest = np.argmax(Ytest, axis=1)
        ytest_ = np.argmax(Ytest_, axis=1)
        labels.extend(ytest.tolist())
        predictions.extend(ytest_.tolist())

    print("test accuracy: {:.3f}".format(accuracy_score(labels, predictions)))
    print("Confusion matrix")
    print(confusion_matrix(labels, predictions))




if __name__ == "__main__":
    main()
