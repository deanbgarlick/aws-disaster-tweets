import argparse
import os

import pandas as pd
import torch

from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification, Trainer, TrainingArguments
from sklearn.model_selection import train_test_split

from dataset import DisasterTweetsDataset


def main(args):

    data = pd.read_csv(os.path.join(args.train, 'data.csv'))

    def preprocess_row(x):
        return x.text + ' [KEYWORD] ' + str(x.keyword) + ' [LOCATION] ' + str(x.location)

    data['tweet'] = data.apply(preprocess_row, axis=1)

    data_tweets, data_labels = data.tweet.astype(str).tolist(), data.target.astype(int).tolist()
    train_tweets, val_tweets, train_labels, val_labels = train_test_split(data_tweets, data_labels, test_size=.2)

    tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')
    model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased")

    train_encodings = tokenizer(train_tweets, truncation=True, padding=True)
    val_encodings = tokenizer(val_tweets, truncation=True, padding=True)

    train_dataset = DisasterTweetsDataset(train_encodings, train_labels)
    val_dataset = DisasterTweetsDataset(val_encodings, val_labels)

    training_args = TrainingArguments(
        output_dir=args.model_dir,      # output directory
        num_train_epochs=args.num_train_epochs,              # total number of training epochs
        per_device_train_batch_size=args.per_device_train_batch_size,  # batch size per device during training
        per_device_eval_batch_size=args.per_device_eval_batch_size,   # batch size for evaluation
        warmup_steps=args.warmup_steps,                # number of warmup steps for learning rate scheduler
        weight_decay=args.weight_decay,               # strength of weight decay
        logging_dir=os.path.join(args.model_dir, '../output'),    # directory for storing logs
        logging_steps=args.logging_steps,
    )

    trainer = Trainer(
        model=model,                         # the instantiated 🤗 Transformers model to be trained
        args=training_args,                  # training arguments, defined above
        train_dataset=train_dataset,         # training dataset
        eval_dataset=val_dataset             # evaluation dataset
    )

    trainer.train()

    tokenizer.save_pretrained(os.path.join(args.model_dir, 'tokenizer'))
    model.save_pretrained(os.path.join(args.model_dir, 'model'))


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--train", type=str, default=os.environ["SM_CHANNEL_TRAIN"])
    parser.add_argument("--model_dir", type=str, default=os.environ["SM_MODEL_DIR"])

    parser.add_argument('--num_train_epochs', type=int, default=3)
    parser.add_argument('--per_device_train_batch_size', type=int, default=16)
    parser.add_argument('--per_device_eval_batch_size', type=int, default=64)
    parser.add_argument('--warmup_steps', type=int, default=500)
    parser.add_argument('--weight_decay', type=int, default=0.01)
    parser.add_argument('--logging_steps', type=int, default=10)

    args = parser.parse_args()

    main(args)
