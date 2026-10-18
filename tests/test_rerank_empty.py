"""Regression test: reranking an empty passage list must return an empty list.

Uses a tiny deterministic ONNX model and a real tokenizers tokenizer written
to a temporary cache directory, so no model download is needed.
"""

import json

import pytest

onnx = pytest.importorskip("onnx")
from onnx import TensorProto, helper
from tokenizers import Tokenizer, models, pre_tokenizers, processors

from flashrank import Ranker, RerankRequest
from flashrank.Config import default_model, model_file_map


def _write_model_dir(model_dir, with_token_type_ids):
    model_dir.mkdir()
    vocab = {"[PAD]": 0, "[UNK]": 1, "[CLS]": 2, "[SEP]": 3, "paris": 4, "berlin": 5}
    tokenizer = Tokenizer(models.WordLevel(vocab, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer.post_processor = processors.TemplateProcessing(
        single="[CLS] $A [SEP]",
        pair="[CLS] $A [SEP] $B:1 [SEP]:1",
        special_tokens=[("[CLS]", 2), ("[SEP]", 3)],
    )
    tokenizer.save(str(model_dir / "tokenizer.json"))
    (model_dir / "config.json").write_text(json.dumps({"pad_token_id": 0}))
    (model_dir / "tokenizer_config.json").write_text(
        json.dumps({"model_max_length": 512, "pad_token": "[PAD]"})
    )
    (model_dir / "special_tokens_map.json").write_text(json.dumps({"pad_token": "[PAD]"}))

    # logits[b, 0] = -sum over the sequence of input_ids * attention_mask (+ token_type_ids)
    inputs = [
        helper.make_tensor_value_info("input_ids", TensorProto.INT64, ["batch", "seq"]),
        helper.make_tensor_value_info("attention_mask", TensorProto.INT64, ["batch", "seq"]),
    ]
    nodes = [helper.make_node("Mul", ["input_ids", "attention_mask"], ["masked"])]
    summed_input = "masked"
    if with_token_type_ids:
        inputs.append(
            helper.make_tensor_value_info("token_type_ids", TensorProto.INT64, ["batch", "seq"])
        )
        nodes.append(helper.make_node("Add", ["masked", "token_type_ids"], ["with_types"]))
        summed_input = "with_types"
    nodes += [
        helper.make_node("ReduceSum", [summed_input, "axes"], ["summed"], keepdims=1),
        helper.make_node("Cast", ["summed"], ["summed_float"], to=TensorProto.FLOAT),
        helper.make_node("Neg", ["summed_float"], ["logits"]),
    ]
    graph = helper.make_graph(
        nodes,
        "tiny_reranker",
        inputs,
        [helper.make_tensor_value_info("logits", TensorProto.FLOAT, ["batch", 1])],
        initializer=[helper.make_tensor("axes", TensorProto.INT64, [1], [1])],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 8
    onnx.save(model, str(model_dir / model_file_map[default_model]))


def _make_ranker(tmp_path, with_token_type_ids):
    _write_model_dir(tmp_path / default_model, with_token_type_ids)
    return Ranker(cache_dir=str(tmp_path))


@pytest.mark.parametrize("with_token_type_ids", [False, True])
def test_rerank_empty_passages_returns_empty_list(tmp_path, with_token_type_ids):
    ranker = _make_ranker(tmp_path, with_token_type_ids)

    assert ranker.rerank(RerankRequest(query="paris", passages=[])) == []
    assert ranker.rerank(RerankRequest(query="paris")) == []


def test_rerank_non_empty_passages_still_scored_and_sorted(tmp_path):
    ranker = _make_ranker(tmp_path, with_token_type_ids=True)
    passages = [{"id": 1, "text": "berlin berlin"}, {"id": 2, "text": "paris"}]

    results = ranker.rerank(RerankRequest(query="paris", passages=passages))

    assert [p["id"] for p in results] == [2, 1]
    assert results[0]["score"] > results[1]["score"] > 0.0
