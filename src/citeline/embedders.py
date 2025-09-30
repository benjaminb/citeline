import numpy as np
import pandas as pd
import torch
from abc import ABC, abstractmethod
from dotenv import load_dotenv
import os
from sentence_transformers import SentenceTransformer
from transformers import AutoModel, AutoTokenizer, AutoModelForCausalLM

"""
Note that some embedders have slightly different operations for embedding queries vs. documents,
so when calling the embedder, specify for_queries=True or False as appropriate.

Usage: how to instantiate and use an Embedder

- Create an embedder via the factory: Embedder.create(model_name, device, normalize, for_queries)
  * model_name: one of the keys returned by list_available_embedders() (e.g. "Qwen/Qwen3-Embedding-0.6B")
  * device: "cuda" | "mps" | "cpu"
  * normalize: bool, whether the returned vectors should be L2-normalized

Example:
    from embedders import Embedder
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.mps.is_available() else "cpu"
    embedder = Embedder.create("Qwen/Qwen3-Embedding-0.6B", device=device, normalize=True)

How to call:
    docs = ["First sentence.", "Second sentence."]
    embeddings = embedder(docs)            # returns a numpy.ndarray of shape (len(docs), dim)
"""
load_dotenv()
DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.mps.is_available() else "cpu"


class Embedder(ABC):
    registry = {}

    @classmethod
    def register(cls, model_name: str):
        def decorator(subclass):
            cls.registry[model_name] = subclass
            return subclass

        return decorator

    @classmethod
    def create(cls, model_name: str, device: str, normalize: bool) -> "Embedder":
        if model_name not in cls.registry:
            available_models = "\n".join(list(cls.registry.keys()))
            raise ValueError(f"Unknown model name: {model_name}. Available models are: {available_models}")
        return cls.registry[model_name](model_name, device, normalize)

    def __init__(self, model_name: str, device, normalize: bool):
        self.model_name = model_name
        self.device = device
        self.normalize = normalize
        self.dim = None

    def __call__(self, docs: list[str] | pd.Series, for_queries: bool = True):
        if isinstance(docs, pd.Series):
            docs = docs.tolist()
        return self._embed(docs, for_queries=for_queries)

    @abstractmethod
    def _embed(self, docs: list[str], for_queries: bool = True) -> np.ndarray:
        pass

    def __str__(self):
        return f"{self.model_name}, device={self.device}, normalize={self.normalize}, dim={self.dim}"


@Embedder.register("AstroMLab/AstroSage-8B")
class AstroSage(Embedder):
    """
    Based on Llama 3.1 8B
    """

    def __init__(self, model_name: str, device: str, normalize: bool):
        # signature order changed to match Embedder.create(...) -> (model_name, device, normalize, for_queries)
        super().__init__(model_name, device, normalize)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModel.from_pretrained(model_name, device_map="auto")
        self.dim = self.model.config.hidden_size
        self.model.eval()

    def _embed(self, docs: list[str]) -> np.ndarray:
        inputs = self.tokenizer(docs, return_tensors="pt", padding=True, truncation=True).to(self.model.device)
        with torch.no_grad():
            outputs = self.model(**inputs, output_hidden_states=True)
        last_hidden_state = outputs.hidden_states[-1]
        embedding = last_hidden_state.mean(dim=1)

        if self.normalize:
            embedding = torch.nn.functional.normalize(embedding, p=2, dim=1)
        return embedding.detach().cpu().numpy()


@Embedder.register("UniverseTBD/astrollama")
class AstroLlamaEmbedder(Embedder):
    """
    Does not use special instruction prompts for embedding queries or docs
    """

    MODEL_DATA = {"max_length": 4096}

    def __init__(self, model_name: str, device: str, normalize: bool):
        super().__init__(model_name, device, normalize)
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


def list_available_embedders() -> list[str]:
    return sorted(Embedder.registry.keys(), key=lambda s: s.lower())


@Embedder.register("Qwen/Qwen3-Embedding-8B")
@Embedder.register("Qwen/Qwen3-Embedding-4B")
@Embedder.register("Qwen/Qwen3-Embedding-0.6B")
class QwenEmbedder(Embedder):
    def __init__(self, model_name: str, device: str, normalize: bool):
        super().__init__(model_name, device, normalize)
        model_kwargs = {}
        # model_kwargs = {"attn_implementation": "flash_attention_2", } if device == "cuda" else {}
        tokenizer_kwargs = {"padding_side": "left"}

        self.model = SentenceTransformer(
            model_name,
            device=device,
            model_kwargs=model_kwargs,
            tokenizer_kwargs=tokenizer_kwargs,
            token=os.environ.get("HUGGINGFACE_API_TOKEN"),
            # cache_folder="~/.cache/huggingface/hub"
        )
        self.model.eval()
        self.dim = self.model.get_sentence_embedding_dimension()

    def _embed(self, docs: list[str], for_queries=True) -> np.ndarray:
        with torch.no_grad():
            kwargs = {"sentences": docs, "show_progress_bar": False}
            if for_queries:
                kwargs["prompt_name"] = "query"
            return self.model.encode(**kwargs)


@Embedder.register("nasa-impact/nasa-ibm-st.38m")
class SentenceTransformerEmbedder(Embedder):
    """
    For a SentenceTransformer based model that does not have separate pipelines for embedding
    docs vs. queries
    """

    def __init__(self, model_name: str, device: str, normalize: bool):
        super().__init__(model_name, device, normalize)
        #
        self.model = SentenceTransformer(model_name, trust_remote_code=True, device=device)
        self.model.eval()
        self.max_length = self.model.get_max_seq_length()  # should be 512

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

    def _embed(self, docs: list[str], for_queries: bool = True) -> np.ndarray:
        """
        This model doesn't have a separate pipeline for queries vs. docs, so ignore for_queries flag
        """
        return self.encode(docs)


@Embedder.register("adsabs/astroBERT")
class AstrobertEmbedder(Embedder):
    MODEL_DATA = {"max_length": 512}

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
        self.max_length = 512

        # NOTE: this works for BERT models but may need adjustment for other architectures
        self.dim = self.model.config.hidden_size

    def _embed(self, docs: list[str], for_queries: bool = True) -> np.ndarray:
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


@Embedder.register("BAAI/bge-small-en")
@Embedder.register("BAAI/bge-large-en-v1.5")
class BGEEmbedder(Embedder):
    INSTRUCTION = "Represent this sentence for searching relevant passages: "

    def __init__(self, model_name: str, device: str, normalize: bool):
        super().__init__(model_name, device, normalize)
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

    def _embed(self, docs: list[str], for_queries: bool = True) -> np.ndarray:
        """
        This model doesn't use a separate pipeline for queries vs. docs, so ignore for_queries flag
        """
        return self.encode(docs)


# NOTE: Specter2 requires adapters modules, which requires transformers ~=4.51.3. I've upgraded
# transformers to 4.56.2 to use gpt-oss models
@Embedder.register("allenai/specter2")
class SpecterEmbedder(Embedder):

    def __init__(self, model_name: str, device: str, normalize: bool, for_queries: bool):
        """
        Because Specter uses different adapters for embedding docs vs. queries, here model_name
        is just the 'key' allenai/specter2. The for_queries flag controls which adapter gets
        loaded:
            allenai/specter2: document embedder
            allenai/specter2_adhoc_query: short query embedder
        """
        super().__init__(model_name, device, normalize, for_queries)
        self.tokenizer = AutoTokenizer.from_pretrained("allenai/specter2_base")
        self.device = device

        # Load the model (adapter gets loaded in _embed method)
        from adapters import AutoAdapterModel

        self.model = AutoAdapterModel.from_pretrained("allenai/specter2_base")

        self.model = self.model.to(self.device)
        self.model.eval()
        self.max_length = 512
        self.dim = self.model.config.hidden_size

    def _embed(self, docs: list[str], for_queries: bool = True) -> np.ndarray:
        adapter_name = "allenai/specter2_adhoc_query" if for_queries else "allenai/specter2"
        self.model.load_adapter(adapter_name, source="hf", set_active=True)
        # Check that adapter is properly loaded onto model
        adapter_code = "[QRY]" if for_queries else "[PRX]"
        assert (
            adapter_code in self.model.active_adapters
        ), f"Specter2 adapter not loaded correctly, {adapter_code} not found in active adapters"
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


def main():
    for name in list_available_embedders():
        print(f"- {name}")
    embedder = Embedder.create(model_name="AstroMLab/AstroSage-8B", device=DEVICE, normalize=True, for_queries=True)
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
