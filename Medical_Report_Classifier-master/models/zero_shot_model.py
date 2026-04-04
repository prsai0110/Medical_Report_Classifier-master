from transformers import pipeline


class ZeroShotClassifier:

    def __init__(self):
        self.classifier = pipeline("zero-shot-classification",
                                   model="facebook/bart-large-mnli")

    def predict(self, text, labels):
        result = self.classifier(text, labels)
        return result['labels'][0]
