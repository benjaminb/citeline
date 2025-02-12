import sys
import numpy as np
import gc
import os
import pytest
import torch
from embedding_functions import sentence_transformer_embedder, encoder_embedder

DEVICE = 'cuda' if torch.cuda.is_available(
) else 'mps' if torch.mps.is_available() else 'cpu'


@pytest.fixture()
def embedder_return_dtype():
    return np.float32


def test_sentence_transformer_embedder_bge(embedder_return_dtype):
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


# @pytest.mark.skipif('DEVICE != "cuda"', reason="CUDA not available")
def test_sentence_transformer_embedder_nvidia():
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
