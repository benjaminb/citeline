import numpy as np
import pandas as pd
import torch
from abc import ABC, abstractmethod

from sentence_transformers import SentenceTransformer
from transformers import AutoModel, AutoTokenizer, AutoModelForCausalLM

DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.mps.is_available() else "cpu"


class Embedder(ABC):
    def __init__(self, model_name: str, device, normalize: bool):
        self.model_name = model_name
        self.device = device
        self.normalize = normalize
        self.dim = None

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

class QwenEmbedder(Embedder):
    MODEL_SPECIFIC_KWARGS = {
        "Qwen/Qwen3-Embedding-8B": {
            # TODO: switch this on for CUDA environment
            # "model_kwargs": {"attn_implementation": "flash_attention_2"}, # Only available on CUDA
            "tokenizer_kwargs": {"padding_side": "left"},
        }
    }
    def __init__(self, model_name: str, device: str, normalize: bool):
        super().__init__(model_name, device, normalize)
        model_kwargs = self.MODEL_SPECIFIC_KWARGS.get(model_name, {}).get("model_kwargs", {})
        tokenizer_kwargs = self.MODEL_SPECIFIC_KWARGS.get(model_name, {}).get("tokenizer_kwargs", {})

        self.model = SentenceTransformer(
            model_name,
            device=device,
            model_kwargs=model_kwargs,
            tokenizer_kwargs=tokenizer_kwargs,
        )
        self.model.eval()
        self.dim = self.model.get_sentence_embedding_dimension()

    def encode(self, docs):
        with torch.no_grad():
            return self.model.encode(
                docs,
                prompt_name="query" # Built-in prompt for better embedding performance
            )

    def _embed(self, docs: list[str]) -> np.ndarray:
        return self.encode(docs)

class SentenceTransformerEmbedder(Embedder):

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

# TODO: move this into class definition, so you can use constructor instead of get_embedder?
EMBEDDING_CLASS = {
    "adsabs/astroBERT": EncoderEmbedder,
    "BAAI/bge-small-en": SentenceTransformerEmbedder,
    "BAAI/bge-large-en-v1.5": SentenceTransformerEmbedder,
    "bert-base-uncased": EncoderEmbedder,
    "nasa-impact/nasa-ibm-st.38m": SentenceTransformerEmbedder,
    "Qwen/Qwen3-Embedding-0.6B": SentenceTransformerEmbedder,
    "Qwen/Qwen3-Embedding-8B": QwenEmbedder,
    "UniverseTBD/astrollama": AstroLlamaEmbedder,
    # astrosage
}

MODEL_DATA = {
    "adsabs/astroBERT": {"max_length": 512},
    "UniverseTBD/astrollama": {"max_length": 4096},
}


def get_embedder(model_name: str, device: str, normalize: bool = False) -> Embedder:
    try:
        return EMBEDDING_CLASS[model_name](model_name, device, normalize)
    except KeyError:
        raise ValueError(f"Model {model_name} not supported. Available models: {list(EMBEDDING_CLASS.keys())}")


def main():
    pass


if __name__ == "__main__":
    main()
