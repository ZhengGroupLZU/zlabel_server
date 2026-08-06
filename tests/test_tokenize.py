"""CLIP BPE tokenizer tests (SAM3 text encoder input semantics)."""

from __future__ import annotations

import numpy as np
import pytest

from app.sam_ort.tokenize import CONTEXT_LENGTH, SimpleTokenizer

VOCAB = "assets/onnx/vocab.json"
MERGES = "assets/onnx/merges.txt"


@pytest.fixture(scope="module")
def tokenizer() -> SimpleTokenizer:
    return SimpleTokenizer(VOCAB, MERGES)


class TestTokenizer:
    def test_bos_eos_structure(self, tokenizer):
        ids = tokenizer("person", context_length=CONTEXT_LENGTH)[0]
        assert ids[0] == tokenizer.encoder["<|startoftext|>"]
        assert ids[1] != tokenizer.encoder["<|endoftext|>"]

    def test_zero_padding(self, tokenizer):
        ids = tokenizer("person", context_length=CONTEXT_LENGTH)[0]
        # padding is 0 (matches the MNN/ONNX text encoder's internal mask)
        assert ids[-1] == 0
        non_zero = ids[ids != 0]
        assert non_zero[-1] == tokenizer.encoder["<|endoftext|>"]

    def test_text_mask_semantics(self, tokenizer):
        ids = tokenizer("person", context_length=CONTEXT_LENGTH)[0]
        mask = (ids == 0).astype(np.int8)[None]
        # True(=1) positions are padding only
        valid = np.where(mask[0] == 0)[0]
        assert len(valid) > 0
        assert mask[0, valid[-1] + 1 :].all()

    def test_truncation(self, tokenizer):
        long_text = " ".join(["seedling"] * 30)
        ids = tokenizer(long_text, context_length=CONTEXT_LENGTH)[0]
        assert len(ids) == CONTEXT_LENGTH
        # truncated text still ends with EOS per SimpleTokenizer contract
        assert ids[CONTEXT_LENGTH - 1] == tokenizer.encoder["<|endoftext|>"]

    def test_case_preserved(self, tokenizer):
        # the numpy tokenizer does not lowercase: different case yields different ids
        upper = tokenizer("PERSON", context_length=CONTEXT_LENGTH)[0]
        lower = tokenizer("person", context_length=CONTEXT_LENGTH)[0]
        assert not np.array_equal(upper, lower)

    def test_multi_word(self, tokenizer):
        ids = tokenizer("green seedling", context_length=CONTEXT_LENGTH)[0]
        # two content words -> more than one non-special token
        content = ids[ids != 0][1:-1]  # strip BOS/EOS
        assert len(content) >= 2

    def test_batch(self, tokenizer):
        out = tokenizer(["person", "seed"], context_length=CONTEXT_LENGTH)
        assert out.shape == (2, CONTEXT_LENGTH)
        assert out.dtype == np.int64
