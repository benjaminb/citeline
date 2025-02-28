import torch
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import AutoModel, AutoTokenizer

DEVICE = 'cuda' if torch.cuda.is_available(
) else 'mps' if torch.mps.is_available() else 'cpu'

class Embedder:
    def __init__(self, model_name: str, device, normalize: bool):
        self.model_name = model_name
        self.device = device
        self.normalize = normalize

    def __str__(self):
        return f"{self.model_name}, normalize={self.normalize}"
        

class SentenceTransformerEmbedder(Embedder):
    def __init__(self, model_name: str, device, normalize: bool):
        super().__init__(model_name, device, normalize)
        self.model = SentenceTransformer(
            model_name, trust_remote_code=True, device=device)
        # self.normalize = normalize

    def __call__(self, docs):
        return self.model.encode(
            docs, convert_to_numpy=True, normalize_embeddings=self.normalize)


class EncoderEmbedder(Embedder):
    def __init__(self, model_name: str, device, normalize: bool):
        super().__init__(model_name, device, normalize)
        # self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(device)
        # self.normalize = normalize
        self.max_length = MODEL_DATA[model_name]['max_length'] if model_name in MODEL_DATA and 'max_length' in MODEL_DATA[model_name] else None

    def __call__(self, docs: list[str]):
        params = {'return_tensors': 'pt', 'padding': True, 'truncation': True}
        if self.max_length:
            params['max_length'] = self.max_length
        inputs = self.tokenizer(docs, **params).to(self.device)
        outputs = self.model(**inputs)
        embeddings = outputs.last_hidden_state[:, 0, :]

        if self.normalize:
            embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)

        return embeddings.detach().cpu().numpy()


EMBEDDING_CLASS = {
    "adsabs/astroBERT": EncoderEmbedder,
    "BAAI/bge-small-en": SentenceTransformerEmbedder,
    "bert-base-uncased": EncoderEmbedder,
    "nvidia/NV-Embed-v2": SentenceTransformerEmbedder
}

MODEL_DATA = {
    "adsabs/astroBERT": {'max_length': 512},
}


def get_embedder(model_name: str, device, normalize: False):
    try:
        return EMBEDDING_CLASS[model_name](model_name, device, normalize)
    except KeyError:
        raise ValueError(
            f"Model {model_name} not supported. Available models: {list(EMBEDDING_CLASS.keys())}")


def main():
    pass


if __name__ == "__main__":
    main()
