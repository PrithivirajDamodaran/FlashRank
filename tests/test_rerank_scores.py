"""Offline regression tests for pairwise reranking score conversion."""

import logging
import unittest
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np

from flashrank import Ranker, RerankRequest


class TestRerankScores(unittest.TestCase):
    def rerank(self, logits):
        """Use real reranking with deterministic tokenizer and model outputs."""
        ranker = Ranker.__new__(Ranker)
        ranker.llm_model = None
        ranker.logger = logging.getLogger(__name__)
        ranker.tokenizer = Mock()
        ranker.tokenizer.encode_batch.return_value = [
            SimpleNamespace(ids=[1, 2], type_ids=[0, 0], attention_mask=[1, 1])
            for _ in logits
        ]
        ranker.session = Mock()
        ranker.session.run.return_value = [logits]
        passages = [{"id": "a", "text": "first"}, {"id": "b", "text": "second"}]
        return ranker.rerank(RerankRequest(query="query", passages=passages))

    def test_softmax_is_invariant_to_row_offsets(self):
        for classes in (2, 3):
            for dtype in (np.float32, np.float64):
                logits = np.array([[0, 1, -1], [0, 2, -1]], dtype=dtype)[:, :classes]
                probabilities = np.exp(logits)[:, 1] / np.exp(logits).sum(axis=1)
                for offsets in ([0, 0], [100, 100], [-100, -100], [1000, -1000]):
                    with self.subTest(classes=classes, dtype=dtype, offsets=offsets):
                        shifted = logits + np.array(offsets, dtype=dtype)[:, None]
                        with np.errstate(over="raise", invalid="raise", divide="raise"):
                            results = self.rerank(shifted)
                        self.assertEqual([row["id"] for row in results], ["b", "a"])
                        scores = np.array([row["score"] for row in results])
                        self.assertTrue(np.isfinite(scores).all())
                        np.testing.assert_allclose(scores, probabilities[::-1], rtol=1e-6)

    def test_single_logit_sigmoid(self):
        results = self.rerank(np.array([[0.0], [1.0]], dtype=np.float32))
        self.assertEqual([row["id"] for row in results], ["b", "a"])
        np.testing.assert_allclose(
            [row["score"] for row in results], [1 / (1 + np.exp(-1)), 0.5], rtol=1e-6
        )


if __name__ == "__main__":
    unittest.main()
