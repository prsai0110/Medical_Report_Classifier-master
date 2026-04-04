from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
import torch
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset

MODEL_NAME = "bert-base-uncased"


class MedicalDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            str(self.texts[idx]),
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            return_tensors="pt"
        )

        return {
            "input_ids": encoding["input_ids"].flatten(),
            "attention_mask": encoding["attention_mask"].flatten(),
            "labels": torch.tensor(self.labels[idx], dtype=torch.long)
        }


class BERTClassifier:

    def __init__(self):
        self.tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)
        self.label_encoder = LabelEncoder()
        self.model = None

    def train(self, csv_path):

        df = pd.read_csv(csv_path)

        texts = df["text"].tolist()
        labels = self.label_encoder.fit_transform(df["label"])

        dataset = MedicalDataset(texts, labels, self.tokenizer)

        num_labels = len(set(labels))

        self.model = BertForSequenceClassification.from_pretrained(
            MODEL_NAME,
            num_labels=num_labels
        )

        training_args = TrainingArguments(
            output_dir="./results",
            num_train_epochs=2,
            per_device_train_batch_size=8,
            logging_steps=10,
            save_strategy="no"
        )

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=dataset
        )

        trainer.train()

        self.model.save_pretrained("models/bert_model")
        self.tokenizer.save_pretrained("models/bert_model")

        print("✅ BERT Model Trained & Saved")

    def load_model(self):
        self.model = BertForSequenceClassification.from_pretrained("models/bert_model")
        self.tokenizer = BertTokenizer.from_pretrained("models/bert_model")

    def predict(self, text):

        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=128
        )

        outputs = self.model(**inputs)
        predicted_class = torch.argmax(outputs.logits, dim=1).item()

        return self.label_encoder.inverse_transform([predicted_class])[0]
