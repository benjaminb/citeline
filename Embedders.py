import torch

# import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import AutoModel, AutoTokenizer

DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.mps.is_available() else "cpu"


class Embedder:

    def __init__(self, model_name: str, device, normalize: bool):
        self.model_name = model_name
        self.device = device
        self.normalize = normalize
        self.dim = None

    def __str__(self):
        return f"{self.model_name}, device={self.device}, normalize={self.normalize}"


class SentenceTransformerEmbedder(Embedder):
    def __init__(self, model_name: str, device: str, normalize: bool):
        super().__init__(model_name, device, normalize)
        self.model = SentenceTransformer(model_name, trust_remote_code=True, device=device)
        self.model.eval()

        # Reattempt multiprocess
        self.pool = None
        if torch.cuda.device_count() > 1:
            self.pool = self.model.start_multi_process_pool(
                target_devices=[f"cuda:{i}" for i in range(torch.cuda.device_count())]
            )

        if self.pool:
            print(f"Using {torch.cuda.device_count()} cuda GPUs for encoding.")

            def encode(docs):
                """
                Create the embedding function in a no_grad context
                """
                with torch.no_grad():
                    return self.model.encode_multi_process(
                        docs,
                        pool=self.pool,
                        normalize_embeddings=self.normalize,
                        show_progress_bar=False,
                    )

        else:

            def encode(docs):
                """
                Create the embedding function in a no_grad context
                """
                with torch.no_grad():
                    return self.model.encode(
                        docs,
                        convert_to_numpy=True,
                        normalize_embeddings=self.normalize,
                        show_progress_bar=False,
                    )

        self.encode = encode

        # Old encoder implementation (no multi process pool)
        # def encode(docs):
        #     """
        #     Create the embedding function in a no_grad context
        #     """
        #     return self.model.encode(
        #         docs,
        #         convert_to_numpy=True,
        #         normalize_embeddings=self.normalize,
        #         show_progress_bar=False)
        # self.encode = encode

        self.dim = self.model.get_sentence_embedding_dimension()

    def __call__(self, docs):
        return self.encode(docs)


class EncoderEmbedder(Embedder):
    def __init__(self, model_name: str, device: str, normalize: bool):
        super().__init__(model_name, device, normalize)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        # Set up the model
        kwargs = {"pretrained_model_name_or_path": model_name, "trust_remote_code": True}

        # If using multiple GPUs, use the 'auto' device map
        if device == "cuda" and torch.cuda.device_count() > 1:
            kwargs["device_map"] = "auto"
            self.model = AutoModel.from_pretrained(**kwargs)
        else:
            model = AutoModel.from_pretrained(**kwargs)
            self.model = model.to(device)

        self.model.eval()
        self.max_length = (
            MODEL_DATA[model_name]["max_length"]
            if model_name in MODEL_DATA and "max_length" in MODEL_DATA[model_name]
            else None
        )

        # NOTE: this works for BERT models but may need adjustment for other architectures
        self.dim = self.model.config.hidden_size

    def __call__(self, docs: list[str]):
        params = {"return_tensors": "pt", "padding": True, "truncation": True}
        if self.max_length:
            params["max_length"] = self.max_length
        inputs = self.tokenizer(docs, **params).to(self.device)
        # Issue warning if input length exceeds model's max_length
        # TODO: use AutoConfig to handle this more gracefully
        if self.max_length and inputs["input_ids"].shape[1] > self.max_length:
            print(
                f"Warning: input length {inputs['input_ids'].shape[1]} exceeds max_length {self.max_length}"
            )

        with torch.no_grad():
            outputs = self.model(**inputs)
        embeddings = outputs.last_hidden_state[:, 0, :]

        if self.normalize:
            embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)

        return embeddings.detach().cpu().numpy()


EMBEDDING_CLASS = {
    "adsabs/astroBERT": EncoderEmbedder,
    "BAAI/bge-small-en": SentenceTransformerEmbedder,
    "BAAI/bge-large-en-v1.5": SentenceTransformerEmbedder,
    "bert-base-uncased": EncoderEmbedder,
    "nasa-impact/nasa-ibm-st.38m": SentenceTransformerEmbedder,
    # "nvidia/NV-Embed-v2": SentenceTransformerEmbedder
}

MODEL_DATA = {
    "adsabs/astroBERT": {"max_length": 512},
}


def get_embedder(model_name: str, device: str, normalize: bool = False) -> Embedder:
    try:
        return EMBEDDING_CLASS[model_name](model_name, device, normalize)
    except KeyError:
        raise ValueError(
            f"Model {model_name} not supported. Available models: {list(EMBEDDING_CLASS.keys())}"
        )


def main():
    pass


if __name__ == "__main__":
    main()
