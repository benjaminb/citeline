import torch
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import AutoModel, AutoTokenizer

DEVICE = 'cuda' if torch.cuda.is_available(
) else 'mps' if torch.mps.is_available() else 'cpu'


class SentenceTransformerEmbedder:
    def __init__(self, model_name: str, device, normalize: bool):
        self.model = SentenceTransformer(
            model_name, trust_remote_code=True, device=device)
        self.normalize = normalize

    def __call__(self, docs):
        return self.model.encode(
            docs, convert_to_numpy=True, normalize_embeddings=self.normalize)


class EncoderEmbedder:
    def __init__(self, model_name: str, device, normalize: bool):
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(device)
        self.normalize = normalize

    def __call__(self, docs: list[str]):
        inputs = self.tokenizer(docs, return_tensors="pt",
                                padding=True, truncation=True).to(self.device)
        outputs = self.model(**inputs)
        embeddings = outputs.last_hidden_state[:, 0, :]

        if self.normalize:
            embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)

        return embeddings.detach().cpu().numpy()


EMBEDDING_CLASS = {
    "BAAI/bge-small-en": SentenceTransformerEmbedder,
    "bert-base-uncased": EncoderEmbedder,
    "nvidia/NV-Embed-v2": SentenceTransformerEmbedder
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
