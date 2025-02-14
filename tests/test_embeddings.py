import numpy as np
import gc
import pytest
import torch
from embedding_functions import sentence_transformer_embedder, encoder_embedder, ChromaEmbedder, get_embedding_fn

DEVICE = 'cuda' if torch.cuda.is_available(
) else 'mps' if torch.mps.is_available() else 'cpu'


# np.float32 | np.int32 required for Chroma
@pytest.fixture()
def embedder_return_dtype():
    return np.float32


@pytest.fixture()
def bge_embedder():
    return sentence_transformer_embedder("BAAI/bge-small-en", DEVICE, normalize=False)


def test_unit_sentence_transformer_embedder_bge(embedder_return_dtype):
    embedder = sentence_transformer_embedder(
        "BAAI/bge-small-en", DEVICE, normalize=False)
    assert embedder is not None

    # Check the output is a function taking a list of strings and returning a numpy array
    input_list = ["a", "I"]
    embeddings = embedder(input_list)

    assert isinstance(embeddings, np.ndarray)
    assert embeddings.shape == (len(input_list), 384)
    assert embeddings.dtype == np.float32
    assert embeddings.dtype == embedder_return_dtype

    # Tear down
    del embedder
    gc.collect()


def test_unit_sentence_transformer_rejects_unknown_model():
    with pytest.raises(ValueError):
        embedder = get_embedding_fn(
            "unknown-model", DEVICE, normalize=False)
        del embedder
        gc.collect()


@pytest.mark.skipif('DEVICE != "cuda"', reason="CUDA not available")
def test_unit_sentence_transformer_embedder_nvidia(embedder_return_dtype):
    embedder = sentence_transformer_embedder(
        "nvidia/NV-Embed-v2", DEVICE, normalize=False)
    assert embedder is not None

    # Check the output is a function taking a list of strings and returning a numpy array
    input_list = ["a", "I"]
    embeddings = embedder(input_list)

    assert isinstance(embeddings, np.ndarray)
    assert embeddings.shape == (len(input_list), 4096)
    assert embeddings.dtype == np.float32
    assert embeddings.dtype == embedder_return_dtype

    # Tear down
    del embedder
    gc.collect()


def test_unit_sentence_transformer_embedder_normalization():
    embedder = sentence_transformer_embedder(
        "BAAI/bge-small-en", DEVICE, normalize=True)
    assert embedder is not None

    input_list = ["a", "I"]
    embeddings = embedder(input_list)

    # Check that the embeddings are normalized
    norms = np.linalg.norm(embeddings, axis=1)
    assert np.allclose(norms, 1.0)

    del embedder
    gc.collect()


@pytest.mark.xfail()
def test_unit_encoder_embedder_unnormalized():
    embedder = encoder_embedder("bert-base-uncased", DEVICE, normalize=False)
    assert embedder is not None

    input_list = ["a", "I"]
    embeddings = embedder(input_list)

    norms = np.linalg.norm(embeddings, axis=1)
    assert not np.allclose(norms, 1.0)


# @pytest.mark.xfail()
def test_unit_sentence_transformer_unnormalized():
    embedder = sentence_transformer_embedder(
        "BAAI/bge-small-en", DEVICE, normalize=False)
    assert embedder is not None

    input_list = ["hello, world",
                  "a long string unlikely to embed to unit vector"]
    embeddings = embedder(input_list)

    norms = np.linalg.norm(embeddings, axis=1)
    assert not np.allclose(norms, 1.0)


def test_unit_encoder_transformer_embedder_normalization():
    embedder = sentence_transformer_embedder(
        "bert-base-uncased", DEVICE, normalize=True)
    assert embedder is not None

    input_list = ["a", "I"]
    embeddings = embedder(input_list)

    # Check that the embeddings are normalized
    norms = np.linalg.norm(embeddings, axis=1)
    assert np.allclose(norms, 1.0)


def test_unit_encoder_embedder_bert(embedder_return_dtype):
    embedder = encoder_embedder("bert-base-uncased", DEVICE, normalize=False)
    assert embedder is not None

    # Check the output is a function taking a list of strings and returning a numpy array
    input_list = ["a", "I"]
    embeddings = embedder(input_list)

    assert isinstance(embeddings, np.ndarray)
    assert embeddings.shape == (len(input_list), 768)
    assert embeddings.dtype == np.float32
    assert embeddings.dtype == embedder_return_dtype

    # Tear down
    del embedder
    gc.collect()


def test_unit_chroma_embedder():
    def embedding_fn(x): return np.random.rand(2, 3).astype(np.float32)
    chroma_embedder = ChromaEmbedder(embedding_fn, "test")
    assert chroma_embedder is not None

    # Check the output is a function taking a list of strings and returning a numpy array
    input_list = ["a", "I"]
    embeddings = chroma_embedder(input_list)

    assert isinstance(embeddings, list)
    assert len(embeddings) == 2
    assert isinstance(embeddings[0], np.ndarray)
    assert embeddings[0].shape == (3,)
    assert embeddings[0].dtype == np.float32


def test_integration_chroma():
    model_name = "BAAI/bge-small-en"
    embedder = sentence_transformer_embedder(
        "BAAI/bge-small-en", DEVICE, normalize=False)
    chroma_embedder = ChromaEmbedder(embedder, model_name)
    assert chroma_embedder is not None

    input_list = ["a", "I"]
    embeddings = chroma_embedder(input_list)

    assert isinstance(embeddings, list)
    assert len(embeddings) == 2
    assert isinstance(embeddings[0], np.ndarray)
    assert embeddings[0].shape == (384,)
    assert embeddings[0].dtype == np.float32
