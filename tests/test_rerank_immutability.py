"""
Tests for rerank() immutability - verifies that rerank() does not mutate the
original passages list or passage dicts passed in via RerankRequest.

Reproduces the bug described in GitHub issue #45.
"""

import copy
import unittest
from unittest.mock import MagicMock, patch
import numpy as np


class MockSession:
    """Minimal ONNX session mock that returns fixed logits."""
    def run(self, output_names, input_feed):
        n = input_feed["input_ids"].shape[0]
        # Return logits shaped (n, 1) so sigmoid path is taken
        return [np.array([[0.5]] * n, dtype=np.float32)]


class MockTokenizer:
    """Minimal tokenizer mock."""
    def encode_batch(self, pairs):
        n = len(pairs)
        seq_len = 4
        result = []
        for _ in range(n):
            enc = MagicMock()
            enc.ids = [0] * seq_len
            enc.type_ids = [0] * seq_len
            enc.attention_mask = [1] * seq_len
            result.append(enc)
        return result


class TestRerankImmutability(unittest.TestCase):
    def _make_ranker(self):
        from flashrank.Ranker import Ranker
        with patch.object(Ranker, "__init__", lambda self, **kw: None):
            ranker = Ranker.__new__(Ranker)
        ranker.llm_model = None
        ranker.session = MockSession()
        ranker.tokenizer = MockTokenizer()
        import logging
        ranker.logger = logging.getLogger("test")
        return ranker

    def _make_request(self):
        from flashrank.Ranker import RerankRequest
        passages = [
            {"id": 1, "text": "Paris is the capital of France."},
            {"id": 2, "text": "Berlin is the capital of Germany."},
            {"id": 3, "text": "Rome is the capital of Italy."},
        ]
        return RerankRequest(query="capital of France", passages=passages), passages

    def test_original_passages_not_modified(self):
        """rerank() must not add 'score' keys to original passage dicts."""
        ranker = self._make_ranker()
        request, original_passages = self._make_request()
        original_copy = copy.deepcopy(original_passages)

        ranker.rerank(request)

        for orig, after in zip(original_copy, original_passages):
            self.assertEqual(orig, after, "Original passage dict was mutated by rerank()")

    def test_original_list_order_preserved(self):
        """rerank() must not reorder the original request.passages list."""
        ranker = self._make_ranker()
        request, original_passages = self._make_request()
        original_ids = [p["id"] for p in original_passages]

        ranker.rerank(request)

        after_ids = [p["id"] for p in original_passages]
        self.assertEqual(original_ids, after_ids, "Original passages list was sorted in-place by rerank()")

    def test_repeated_calls_independent(self):
        """Successive rerank() calls on the same request must return independent results."""
        ranker = self._make_ranker()
        request, _ = self._make_request()

        result1 = ranker.rerank(request)
        result2 = ranker.rerank(request)

        # Mutating result1 must not affect result2
        result1[0]["score"] = -999.0
        self.assertNotEqual(result2[0].get("score"), -999.0,
                            "Returned lists share mutable references across calls")

    def test_result_contains_scores(self):
        """rerank() must return passages with 'score' added and sorted descending."""
        ranker = self._make_ranker()
        request, _ = self._make_request()

        results = ranker.rerank(request)

        self.assertEqual(len(results), 3)
        scores = [r["score"] for r in results]
        self.assertEqual(scores, sorted(scores, reverse=True))


if __name__ == "__main__":
    unittest.main()
