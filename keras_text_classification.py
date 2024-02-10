# The configuration achieves +- 90% accuracy on test set 

# 0. imports and logging config
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''

from tensorflow import keras
import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow.keras import layers
from datasets import load_dataset
from gensim.models import KeyedVectors
import numpy as np
import logging
from rich.logging import RichHandler

# Rich Handler for colorized logging, you can safely remove it
logging.basicConfig(
    level="INFO",
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(rich_tracebacks=True)],
)

logger = logging.getLogger(__name__)

# ---------------- #
# Optional TODO: If you want to use any additional imports, add them here
# ---------------- #

# YOUR CODE HERE

from tensorflow.strings import regex_replace # for string formatting
import gensim.downloader as api

# ---------------- #

# 1. load dataset
# ---------------- #
# TODO(required, no points): Choose the dataset and uncomment line with dataset you want to use
# ---------------- #

dataset_config = ("emotion", "split", "text", "label") # Emotion Classification
#dataset_config = ("dbpedia_14", None, "content", "label") # DBpedia 14 Classification
#dataset_config = ("amazon_reviews_multi", "en", "review_body", "stars") # Amazon Reviews Multi Language Classification - English

dataset_dict = load_dataset(*dataset_config[:2])
train_dataset = dataset_dict["train"]
validation_dataset = dataset_dict["validation"]
test_dataset = dataset_dict["test"]

# Reduce dataset size for faster training
if len(train_dataset) > 5000:
    train_dataset = train_dataset.shuffle(seed=42)[:5000]

logger.info(f"Train dataset size: {len(train_dataset)}")

if len(validation_dataset) > 500:
    validation_dataset = validation_dataset.shuffle(seed=42)[:500]

TEXT_COLUMN = dataset_config[2]
LABEL_COLUMN = dataset_config[3]

labels = list(set(train_dataset[LABEL_COLUMN]))
label_map = {
    v: i for i, v in enumerate(labels)
}

# 2. preprocessing texts

# ---------------- #
# Before creating vocabulary and vectorizing texts, you need to preprocess them.
# TODO: Implement preprocessing function, which will be used in TextVectorization layer.
# For instance - you can use lowercasing, remove punctuation, remove stopwords, etc.
# Remebert that different datasets have different preprocessing requirements. Explore the dataset!
# Hint: input_data is not a string, but a tensor of strings. Therefore, you need to use functions from tf.strings
# See docs: https://www.tensorflow.org/api_docs/python/tf/strings
# ---------------- #

def preprocess_text(input_data):
    # YOUR CODE HERE

    input_data = tf.strings.lower(input_data) 
    input_data = regex_replace(input_data, "[^a-zA-Z\s]", "") # everything that is not (^) letters or space (\s) will be replaced by "" hence removed
    
    stopwords = ["the", "is", "a", "in", "at", "to", "when", "where"]
    for stopword in stopwords:
        input_data = regex_replace(input_data, f"\\b{stopword}\\b", "")

    return input_data

# ---------------- #

text_vectorizer = layers.TextVectorization(
    max_tokens=30000,
    standardize=preprocess_text,
    output_sequence_length=256
)

text_vectorizer.adapt(train_dataset[TEXT_COLUMN])

logger.info(f"Vocabulary size: {len(text_vectorizer.get_vocabulary())}")
logger.info(f"Vocabulary content: {text_vectorizer.get_vocabulary()[:10]}")

voc = text_vectorizer.get_vocabulary()
word_index = dict(zip(voc, range(len(voc))))

# 3. load pre-trained word embeddings
def load_word_embeddings() -> KeyedVectors:
    # ---------------- #
    # TODO: Implement loading word embeddings from file.
    # You can use gensim library to load embeddings.
    # See docs: https://radimrehurek.com/gensim/models/keyedvectors.html
    # ---------------- #
    embeddings = api.load('word2vec-google-news-300')
    # YOUR CODE HERE

    # ---------------- #
    logger.info(f"Loaded {embeddings} word vectors.")
    return embeddings

def create_embedding_matrix(word_index, embeddings_index):
    embedding_matrix = np.zeros((len(word_index) + 1, embeddings_index.vector_size))
    for word, i in word_index.items():
        if word in embeddings_index:
            embedding_matrix[i] = embeddings_index[word]
    return embedding_matrix

embeddings = load_word_embeddings()
embeddings_matrix = create_embedding_matrix(word_index, embeddings)

logger.info(f"Embeddings matrix shape: {embeddings_matrix.shape}")
logger.info(f"Embeddings matrix: {embeddings_matrix[:2, :3]}")

# 4. create model

def create_model(embedding_matrix: np.ndarray, num_classes: int) -> keras.Model:

    input_layer = layers.Input(shape=(None,), dtype=tf.int64, name="input_layer")
    embedding_layer = layers.Embedding(
        input_dim=embedding_matrix.shape[0],
        output_dim=embedding_matrix.shape[1],
        weights=[embedding_matrix],
        trainable=False,
        name="embedding_layer",
    )(input_layer)

    # ---------------- #
    # TODO: Implement model architecture.
    # You can use any layers you want, but you need to use at least one of the following:
    # - GRU
    # - LSTM
    # - Bidirectional + Any of the above
    # See docs on how to use RNN layers for sequence classification:
    # https://keras.io/guides/working_with_rnns/
    # ---------------- #

    
    # YOUR CODE HERE

    # ---------------- #
    
    rnn_layer = layers.Bidirectional(layers.LSTM(64, return_sequences=True))(embedding_layer)

    # Global Max Pooling Layer
    pooling_layer = layers.GlobalMaxPooling1D()(rnn_layer)

    # Dense Hidden Layer
    dense_layer = layers.Dense(64, activation='relu')(pooling_layer)

    # Output Layer
    output= layers.Dense(num_classes, activation="softmax", name="output_layer")(dense_layer)
    
    model = keras.Model(inputs=input_layer, outputs=output)
    return model


model = create_model(embeddings_matrix, len(set(train_dataset[LABEL_COLUMN])))

model.summary()

# 5. train model
x_train = text_vectorizer(np.array([s for s in train_dataset[TEXT_COLUMN]])).numpy()
y_train = np.array([label_map[l] for l in train_dataset[LABEL_COLUMN]])

x_val = text_vectorizer(np.array([s for s in validation_dataset[TEXT_COLUMN]])).numpy()
y_val = np.array([label_map[l] for l in validation_dataset[LABEL_COLUMN]])

# ---------------- #
# TODO: Compile and train your model. Choose appropriate loss function and optmizer. For metrics use categorical accuracy score.
# See docs: https://www.tensorflow.org/api_docs/python/tf/keras/Model#compile
# See docs: https://www.tensorflow.org/api_docs/python/tf/keras/Model#fit
# ---------------- #

# YOUR CODE HERE

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['sparse_categorical_accuracy'])

history = model.fit(x_train, y_train,
                    epochs=5,  
                    validation_data=(x_val, y_val))

# ---------------- #

# 6. evaluate model
x_test = text_vectorizer(np.array([s for s in test_dataset[TEXT_COLUMN]])).numpy()
y_test = np.array([label_map[l] for l in test_dataset[LABEL_COLUMN]])

results = model.evaluate(x_test, y_test)

# ---------------- #
for name, value in zip(model.metrics_names, results):
    logger.info(f"{name}: {value}")

logger.info("Done!")
