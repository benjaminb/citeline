from sentence_transformers import SentenceTransformer
from transformers import AutoModel, AutoTokenizer
import torch
import numpy as np
import chromadb
from chromadb import Documents, EmbeddingFunction, Embeddings


DEVICE = 'cuda' if torch.cuda.is_available(
) else 'mps' if torch.mps.is_available() else 'cpu'


class ChromaEmbedder(EmbeddingFunction):
    def __init__(self, embedding_fn):
        self._encode = embedding_fn

    def __call__(self, input: Documents) -> Embeddings:
        return self._encode(input)


def sentence_transformer_embedder(model_name: str, device, normalize: False) -> ChromaEmbedder:
    """

    """
    model = SentenceTransformer(
        model_name, trust_remote_code=True, device=device)

    def embedding_lambda(docs): return model.encode(
        docs, convert_to_numpy=True, normalize_embeddings=normalize)
    return ChromaEmbedder(embedding_lambda)


def encoder_embedder(model_name: str, device, normalize: False) -> ChromaEmbedder:
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device)

    def embedder(sentences: list[str]):
        inputs = tokenizer(sentences, return_tensors="pt",
                           padding=True, truncation=True).to(device)
        outputs = model(**inputs)
        embeddings = outputs.last_hidden_state[:, 0, :]
        if normalize:
            embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
        # Convert to list of numpy arrays (for compatibility with Chroma)
        embeddings = embeddings.detach().cpu().numpy().tolist()
        return [np.array(embedding) for embedding in embeddings]

    return ChromaEmbedder(embedder)


EMBEDDING_FN = {
    "BAAI/bge-small-en": sentence_transformer_embedder,
    "bert-base-uncased": encoder_embedder
}


def get_embedding_fn(model_name: str, device, normalize: False) -> ChromaEmbedder:
    if model_name in EMBEDDING_FN:
        return EMBEDDING_FN[model_name](model_name, device, normalize)
    else:
        raise ValueError(f"Model {model_name} not supported")


def main():
    model_name = "bert-base-uncased"
    embedder = get_embedding_fn(model_name=model_name,
                                device=DEVICE,
                                normalize=True)

    result = embedder(["Hello, world!", "How are you?"])
    print(len(result))


if __name__ == "__main__":
    main()
