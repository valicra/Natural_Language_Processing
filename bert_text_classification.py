# The configuration achieves +- 90% accuracy on test set 

# 0. imports and logging config
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

from tensorflow import keras
import tensorflow as tf
from datasets import load_dataset
import transformers as tr
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
from transformers import AutoTokenizer
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.metrics import SparseCategoricalAccuracy

# ---------------- #

# 1. load dataset
# ---------------- #
# TODO(required, no points): Choose the dataset and uncomment line with dataset you want to use
# ---------------- #
dataset_config = ("emotion", "split", "text", "label") # Emotion Classification
#dataset_config = ("dbpedia_14", None, "content", "label") # DBpedia 14 Classification
#dataset_config = ("amazon_reviews_multi", "en", "review_body", "stars") # Amazon Reviews Multi Language Classification - English

dataset_dict = load_dataset(*dataset_config[:2])

TEXT_COLUMN = dataset_config[2]
LABEL_COLUMN = dataset_config[3]

# ---------------- #

# 2. load pre-trained tokenizer
# ---------------- #
# TODO: Choose a model from hugginface hub and load pre-trained tokenizer for that model
# Note that different models may require significantly different amount of computational resources, as well as yield different prediction quality
# Use models listed here: https://huggingface.co/models?pipeline_tag=fill-mask
# You should also check if chosen model has tensorflow implementation for sequence classification.
# Hint: try Auto Classes - https://huggingface.co/docs/transformers/v4.26.0/en/model_doc/auto
# ---------------- #

# YOUR CODE HERE
MODEL_NAME = 'bert-base-uncased'
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)


# 3. text tokenization & data preparation

# ---------------- #
# Transformer models use special tokenizers, which can split rare words into more frequently observed pieces
# TODO: implement tokenize_fn, that will use pre-trained tokenizer to convert strings of input data into dictionaries of token ids and attention masks
# Note that this function must be able to process batches of text. It should also truncate sequences of tokens that are too long for the model.
# See docs for tokenizer class: https://huggingface.co/docs/transformers/v4.26.0/en/main_classes/tokenizer
# ---------------- #

def tokenize_fn(examples):
    # YOUR CODE HERE
    tokenized_inputs = tokenizer(
        examples["text"],
        max_length=256,  # Set the maximum length of the sequence as per your model's requirements
        padding="max_length",  # Pad or truncate the sequences to the specified max length
        truncation=True,  # Truncate sequences that exceed the max length
        return_tensors="tf",  # tf tensors
    )

    return tokenized_inputs

# ---------------- #

dataset_dict = dataset_dict.map(tokenize_fn, batched=True)

# Cleaning and renaming columns
dataset_dict = dataset_dict.rename_column(LABEL_COLUMN, "label")
column_names = dataset_dict.column_names['train']
redundant_column_names = set(column_names) - {'input_ids', 'attention_mask', 'label'}
dataset_dict = dataset_dict.remove_columns(list(redundant_column_names))

id2label = list(set(dataset_dict['train']['label']))
label2id = {
    v: i for i, v in enumerate(id2label)
}

def remap_labels(example):
    return {'label': label2id[example['label']]}

dataset_dict = dataset_dict.map(remap_labels)

logger.info(f"Dataset: {dataset_dict}")

train_dataset = dataset_dict["train"]
validation_dataset = dataset_dict["validation"]
test_dataset = dataset_dict["test"]

# Reduce dataset size for faster training
if len(train_dataset) > 5000:
    train_dataset = train_dataset.shuffle(seed=42).select(range(5000))

logger.info(f"Train dataset size: {len(train_dataset)}")

if len(validation_dataset) > 500:
    validation_dataset = validation_dataset.shuffle(seed=42).select(range(500))


# Data Collator is needed to ensure that batches of texts are aligned and padded to the same length
data_collator = tr.DataCollatorWithPadding(tokenizer=tokenizer, return_tensors="tf")

# 4. Preparing optimizer
batch_size = 16

# ---------------- #
# TODO: Set up hyperparameters for the model training
# Different LR and number of epochs will result in different evaluation metrics. Try various options.

# YOUR CODE HERE
num_epochs = 3
learning_rate = 2e-5
# ---------------- #

batches_per_epoch = len(train_dataset) // batch_size
total_train_steps = int(batches_per_epoch * num_epochs)
optimizer, schedule = tr.create_optimizer(init_lr=2e-5, num_warmup_steps=0, num_train_steps=total_train_steps)

# 5. Load model
# Loading model that you have chosen on step 2

model = tr.TFAutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME, num_labels=len(id2label), id2label=id2label, label2id=label2id
)

# 6. Preparing datasets

logger.info(f"Train dataset: {train_dataset}")

tf_train_dataset = model.prepare_tf_dataset(
     train_dataset,
     shuffle=True,
     batch_size=batch_size,
     collate_fn=data_collator,
)

tf_val_dataset = model.prepare_tf_dataset(
     validation_dataset,
     shuffle=True,
     batch_size=batch_size,
     collate_fn=data_collator,
)

# 7. Model training
# ---------------- #
# TODO: compile and train model, similarly to previous task. Use prepared training and validation dataset, as well as the optimizer. Use categorical accuracy as metric.
# ---------------- #

# YOUR CODE HERE

model.compile(
    optimizer=Adam(learning_rate=learning_rate),
    loss=SparseCategoricalCrossentropy(from_logits=True),
    metrics=[SparseCategoricalAccuracy(name='accuracy')],
)

# Train the model
history = model.fit(
    tf_train_dataset,
    validation_data=tf_val_dataset,
    epochs=num_epochs,
)


# ---------------- #

# 6. evaluate model

tf_test_dataset = model.prepare_tf_dataset(
     test_dataset,
     shuffle=True,
     batch_size=batch_size,
     collate_fn=data_collator,
)

results = model.evaluate(tf_test_dataset)

# ---------------- #
for name, value in zip(model.metrics_names, results):
    logger.info(f"{name}: {value}")

logger.info("Done!")
