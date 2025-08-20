import numpy as np
import pandas as pd
import torch
from abc import ABC, abstractmethod

from sentence_transformers import SentenceTransformer
from transformers import AutoModel, AutoTokenizer, AutoModelForCausalLM

DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.mps.is_available() else "cpu"


class Embedder(ABC):
    def __init__(self, model_name: str, device, normalize: bool, for_queries: bool):
        self.model_name = model_name
        self.device = device
        self.normalize = normalize
        self.dim = None
        self.for_queries = for_queries

    def __call__(self, docs: list[str] | pd.Series):
        if isinstance(docs, pd.Series):
            docs = docs.tolist()
        return self._embed(docs)

    @abstractmethod
    def _embed(self, docs: list[str]) -> np.ndarray:
        pass

    def __str__(self):
        return f"{self.model_name}, device={self.device}, normalize={self.normalize}"


class AstroLlamaEmbedder(Embedder):
    """
    Does not use special instruction prompts for embedding queries or docs
    """

    def __init__(self, model_name: str, device: str, normalize: bool, for_queries: bool = False):
        super().__init__(model_name, device, normalize, for_queries)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")
        self.model.eval()
        self.dim = self.model.config.hidden_size
        self.max_length = self.model.config.max_position_embeddings

    def _embed(self, docs: list[str]) -> np.ndarray:
        params = {
            "return_tensors": "pt",
            "return_token_type_ids": False,
            "padding": True,
            "truncation": True,
            "max_length": self.max_length,
        }
        # TODO: if we can put the inputs on the same device as the model, we don't need a self.device attribute
        inputs = self.tokenizer(docs, **params).to(self.model.device)
        with torch.no_grad():
            outputs = self.model(**inputs, output_hidden_states=True)
        embeddings = outputs["hidden_states"][-1][:, 1:, ...].mean(dim=1)

        if self.normalize:
            embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
        return embeddings.detach().cpu().numpy()


class QwenEmbedder(Embedder):
    def __init__(self, model_name: str, device: str, normalize: bool, for_queries: bool):
        super().__init__(model_name, device, normalize, for_queries)
        # model_kwargs = {"attn_implementation": "flash_attention_2"} if device == "cuda" else {}
        model_kwargs = {}
        tokenizer_kwargs = {"padding_side": "left"}

        self.model = SentenceTransformer(
            model_name,
            device=device,
            model_kwargs=model_kwargs,
            tokenizer_kwargs=tokenizer_kwargs,
        )
        self.model.eval()
        self.dim = self.model.get_sentence_embedding_dimension()

    def _embed(self, docs: list[str]) -> np.ndarray:
        with torch.no_grad():
            kwargs = {"sentences": docs}
            if self.for_queries:
                kwargs["prompt_name"] = "query"
            return self.model.encode(**kwargs)


class SentenceTransformerEmbedder(Embedder):
    """
    For a SentenceTransformer based model that does not have separate pipelines for embedding
    docs vs. queries
    """

    def __init__(self, model_name: str, device: str, normalize: bool, for_queries: bool = False):
        super().__init__(model_name, device, normalize, for_queries)
        #
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
        self.dim = self.model.get_sentence_embedding_dimension()

    def _embed(self, docs: list[str]) -> np.ndarray:
        return self.encode(docs)


class AstrobertEmbedder(Embedder):
    def __init__(self, model_name: str, device: str, normalize: bool, for_queries: bool = False):
        super().__init__(model_name, device, normalize, for_queries)
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

    def _embed(self, docs: list[str]) -> np.ndarray:
        params = {"return_tensors": "pt", "padding": True, "truncation": True}
        if self.max_length:
            params["max_length"] = self.max_length
        inputs = self.tokenizer(docs, **params).to(self.device)
        # Issue warning if input length exceeds model's max_length
        # TODO: use AutoConfig to handle this more gracefully
        if self.max_length and inputs["input_ids"].shape[1] > self.max_length:
            print(f"Warning: input length {inputs['input_ids'].shape[1]} exceeds max_length {self.max_length}")

        with torch.no_grad():
            outputs = self.model(**inputs)
        embeddings = outputs.last_hidden_state[:, 0, :]

        if self.normalize:
            embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)

        return embeddings.detach().cpu().numpy()


class BGEEmbedder(Embedder):
    INSTRUCTION = "Represent this sentence for searching relevant passages: "

    def __init__(self, model_name: str, device: str, normalize: bool, for_queries: bool = False):
        super().__init__(model_name, device, normalize, for_queries)
        #
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
                if self.for_queries:
                    docs = [self.INSTRUCTION + doc for doc in docs]
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
                if self.for_queries:
                    docs = [self.INSTRUCTION + doc for doc in docs]
                with torch.no_grad():
                    return self.model.encode(
                        docs,
                        convert_to_numpy=True,
                        normalize_embeddings=self.normalize,
                        show_progress_bar=False,
                    )

        self.encode = encode
        self.dim = self.model.get_sentence_embedding_dimension()

    def _embed(self, docs: list[str]) -> np.ndarray:
        return self.encode(docs)


class SpecterEmbedder(Embedder):
    ADAPTER_MAP = {
        "allenai/specter2": ("allenai/specter2", "[PRX]"),
        "allenai/specter2_adhoc_query": ("allenai/specter2_adhoc_query", "[QRY]"),
    }

    def __init__(self, model_name: str, device: str, normalize: bool, for_queries: bool):
        """
        Because Specter uses different adapters for embedding docs vs. queries, here model_name
        actually refers to the adapter
            allenai/specter2: document embedder
            allenai/specter2_adhoc_query: short query embedder
        """
        super().__init__(model_name, device, normalize, for_queries)
        assert for_queries == (
            model_name == "allenai/specter2_adhoc_query"
        ), "If for_queries is True, model_name should be 'allenai/specter2_adhoc_query'. Otherwise, it should be 'allenai/specter2'."

        self.tokenizer = AutoTokenizer.from_pretrained("allenai/specter2_base")
        self.device = device
        self.adapter_name = model_name

        # Load the model and adapter
        from adapters import AutoAdapterModel

        self.model = AutoAdapterModel.from_pretrained("allenai/specter2_base")
        self.model.load_adapter(self.adapter_name, source="hf", set_active=True)

        # Check that adapter is properly loaded onto model
        adapter_code = "[QRY]" if for_queries else "[PRX]"
        assert (
            adapter_code in self.model.active_adapters
        ), f"Specter2 adapter not loaded correctly, {adapter_code} not found in active adapters"

        self.model = self.model.to(self.device)
        self.model.eval()
        self.max_length = 512
        self.dim = self.model.config.hidden_size

    def _embed(self, docs: list[str]) -> np.ndarray:
        params = {
            "padding": True,
            "truncation": True,
            "return_tensors": "pt",
            "max_length": 512,
            "return_token_type_ids": False,
        }
        inputs = self.tokenizer(docs, **params).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)
        embeddings = outputs.last_hidden_state[:, 0, :]

        if self.normalize:
            embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)

        return embeddings.detach().cpu().numpy()


# TODO: move this into class definition, so you can use constructor instead of get_embedder?
EMBEDDING_CLASS = {
    "adsabs/astroBERT": AstrobertEmbedder,
    "BAAI/bge-small-en": BGEEmbedder,
    "BAAI/bge-large-en-v1.5": BGEEmbedder,
    "nasa-impact/nasa-ibm-st.38m": SentenceTransformerEmbedder,
    "Qwen/Qwen3-Embedding-0.6B": QwenEmbedder,
    "Qwen/Qwen3-Embedding-8B": QwenEmbedder,
    "UniverseTBD/astrollama": AstroLlamaEmbedder,
    "allenai/specter2": SpecterEmbedder,  # Use this for embedding documents
    "allenai/specter2_adhoc_query": SpecterEmbedder,  # Use this for embedding queries
    # astrosage
}

MODEL_DATA = {
    "adsabs/astroBERT": {"max_length": 512},
    "UniverseTBD/astrollama": {"max_length": 4096},
}


def get_embedder(model_name: str, device: str, normalize: bool, for_queries: bool) -> Embedder:
    try:
        return EMBEDDING_CLASS[model_name](model_name, device, normalize, for_queries)
    except KeyError:
        raise ValueError(f"Model {model_name} not supported. Available models: {list(EMBEDDING_CLASS.keys())}")


def main():
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.mps.is_available() else "cpu"
    embedder = get_embedder("Qwen/Qwen3-Embedding-0.6B", device=device, normalize=False)
    print(f"Loaded model: {embedder}")
    sample_docs = [
        "This is a test document.",
        "Another document for testing purposes.",
        "Yet another example of a document to embed.",
    ]
    embeddings = embedder(sample_docs)
    print(f"Embeddings shape: {embeddings.shape}")
    print(f"Embeddings norms: {np.linalg.norm(embeddings, axis=1)}")


if __name__ == "__main__":
    main()
