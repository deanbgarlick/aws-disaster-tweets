import torch


class ModelWrapper:

    def __init__(self, tokenizer, classifier):
        self._tokenizer = tokenizer
        self._classifier = classifier
        self._device = self._classifier.device

    def __call__(self, text):
        inputs = self._tokenizer.encode_plus(text, return_tensors = "pt")
        with torch.no_grad():
            outputs = self._classifier(**inputs)
            output = torch.argmax(outputs.logits).item()
        return output
