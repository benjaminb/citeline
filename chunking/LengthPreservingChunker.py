from langchain_experimental.text_splitter import SemanticChunker
import re


class LengthPreservingChunker(SemanticChunker):
    def __init__(
        self,
        embeddings,
        buffer_size: int = 1,
        breakpoint_threshold_type: str = "percentile",
        breakpoint_threshold_amount: float = 95,
        number_of_chunks: int = None,
        sentence_split_regex: str = r"(?<=[.?!]\s)(?=\S)",
        min_chunk_size: int = 64,
    ):
        super().__init__(
            embeddings=embeddings,
            buffer_size=buffer_size,
            breakpoint_threshold_type=breakpoint_threshold_type,
            breakpoint_threshold_amount=breakpoint_threshold_amount,
            number_of_chunks=number_of_chunks,
            sentence_split_regex=sentence_split_regex,
            min_chunk_size=min_chunk_size,
        )
        # Warn user if regex consumes chars
        zero_width_pattern = re.compile(
            r"""^(?:
            \(\?<=[^)]*\)   # positive lookbehind
            | \(\?<![^)]*\)   # negative lookbehind
            | \(\?=[^)]*\)    # positive lookahead
            | \(\?![^)]*\)    # negative lookahead
            | [\^$]           # start/end anchors
            | \\[bBAZz]       # \b, \B, \A, \Z, \z
            )*$""",
            re.VERBOSE,
        )
        if not zero_width_pattern.match(sentence_split_regex):
            print(
                "Warning: The sentence_split_regex pattern may consume characters. "
                "This may modify total text length after splitting text."
            )

    def split_text(self, text: str) -> list[str]:
        # Warn user if regex consumes chars

        single_sentences = re.split(self.sentence_split_regex, text)
        # 2. everything else the same up through finding breakpoints...
        distances, sentences = self._calculate_sentence_distances(single_sentences)
        if self.number_of_chunks is None:
            threshold, dist_array = self._calculate_breakpoint_threshold(distances)
        else:
            threshold = self._threshold_from_clusters(distances)
            dist_array = distances

        breakpoints = {i for i, d in enumerate(dist_array) if d > threshold}

        # 3. build your chunks **without** injecting extra spaces
        chunks = []
        start = 0
        for bp in sorted(breakpoints):
            group = sentences[start : bp + 1]
            # ← ← ← here’s the only change
            combined = "".join(d["sentence"] for d in group)
            chunks.append(combined)
            start = bp + 1

        # last tail
        if start < len(sentences):
            tail = "".join(d["sentence"] for d in sentences[start:])
            chunks.append(tail)

        return chunks
