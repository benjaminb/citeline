import torch
from Embedders import sentence_transformer_embedder, encoder_embedder

DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.mps.is_available() else "cpu"


def test_embedding_bge(benchmark):
    embedder = sentence_transformer_embedder(
        model_name="BAAI/bge-small-en", device=DEVICE, normalize=False
    )
    benchmark(
        embedder,
        "A study of the brightest 55 clusters ( Figure 3 ) ( Dunn Fabian 2006 , 2008 ) originally showed that over 70% of those clusters where the cooling time is less than 3 Gyr, and that therefore need heat, have bubbles; the remaining 30% have a central radio source.",
    )


def test_embedding_bert(benchmark):
    embedder = encoder_embedder(model_name="bert-base-uncased", device=DEVICE, normalize=False)
    benchmark(
        embedder,
        [
            "A study of the brightest 55 clusters ( Figure 3 ) ( Dunn Fabian 2006 , 2008 ) originally showed that over 70% of those clusters where the cooling time is less than 3 Gyr, and that therefore need heat, have bubbles; the remaining 30% have a central radio source."
        ],
    )


def test_embedding_nvidia(benchmark):
    device = "cpu" if DEVICE == "mps" else DEVICE  # NV-Embed-2 incompatible with MPS
    embedder = sentence_transformer_embedder(
        model_name="nvidia/NV-Embed-v2", device=device, normalize=False
    )
    benchmark(
        embedder,
        [
            "A study of the brightest 55 clusters ( Figure 3 ) ( Dunn Fabian 2006 , 2008 ) originally showed that over 70% of those clusters where the cooling time is less than 3 Gyr, and that therefore need heat, have bubbles; the remaining 30% have a central radio source."
        ],
    )
