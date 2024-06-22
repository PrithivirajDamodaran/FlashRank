
<img src=./images/logo.png width=100%>


[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.11093524.svg)](https://doi.org/10.5281/zenodo.11093524)

## [IMPORTANT UPDATE]

~~*A clone library called **SwiftRank is pointing to our model buckets, we are working on a interim solution to avoid this stealing**. Thank you for patience and understanding.*~~

This issue is resolved, the models are in HF now. **please upgrade to continue** pip install -U flashrank. Thank you for patience and understanding

# 🏎️ What is it?
Ultra-lite &amp; Super-fast Python library to add re-ranking to your existing search &amp; retrieval pipelines. It is based on SoTA LLMs and cross-encoders, with gratitude to all the model owners. 

Supports:

- Pairwise / Pointwise rerankers. (Cross encoder based)
- Listwise LLM based rerankers. (LLM based)
(see below for full list of supported models)

# Table of Contents  

1. [Features](#features)  
2. [Installation](#installation)
3. [Getting started](#getting-started)
4. [Deployment patterns](#deployment-patterns)
5. [How to Cite?](#how-to-cite)
5. [Papers citing flashrank](#papers-citing-flashrank) 


## Features

1. ⚡ **Ultra-lite**: 
    - **No Torch or Transformers** needed. Runs on CPU.
    - Boasts the **tiniest reranking model in the world, ~4MB**.
    
2. ⏱️ **Super-fast**:
    - Rerank speed is a function of **# of tokens in passages, query + model depth (layers)**
    - To give an idea, Time taken by the example (in code) using the default model is below.
    - <center><img src="./images/time.png" width=600/></center>
    - Detailed benchmarking, TBD

3. 💸 **$ concious**:
    - **Lowest $ per invocation:** Serverless deployments like Lambda are charged by memory & time per invocation*
    - **Smaller package size** = shorter cold start times, quicker re-deployments for Serverless.

4. 🎯 **Based on SoTA Cross-encoders and other models**:
    - How good are Zero-shot rerankers - look at the reference section.
    - Below are the list of models supported as of now.
        * `ms-marco-TinyBERT-L-2-v2` (default) [Model card](https://huggingface.co/cross-encoder/ms-marco-TinyBERT-L-2)
        * `ms-marco-MiniLM-L-12-v2` [Model card](https://huggingface.co/cross-encoder/ms-marco-MiniLM-L-12-v2)
        * `rank-T5-flan` (Best non cross-encoder reranker) [Model card](https://huggingface.co/bergum/rank-T5-flan)
        * `ms-marco-MultiBERT-L-12`  (Multi-lingual, [supports 100+ languages](https://github.com/google-research/bert/blob/master/multilingual.md#list-of-languages))
        * `ce-esci-MiniLM-L12-v2` [FT on Amazon ESCI dataset](https://github.com/amazon-science/esci-data) (This is interesting because most models are FT on MSFT MARCO Bing queries) [Model card](https://huggingface.co/metarank/ce-esci-MiniLM-L12-v2)
        * `rank_zephyr_7b_v1_full` (4-bit-quantised GGUF) [Model card](https://huggingface.co/castorini/rank_zephyr_7b_v1_full) (Offers very competitive performance, with large context window and relatively faster for a 4GB model).
            - **Important note:** Our current integration of `rank_zephyr` supports a max of 20 passages in one pass. The sliding window logic support is yet to be added.
    - Models in roadmap:
        * InRanker
    - Why sleeker models are preferred ? Reranking is the final leg of larger retrieval pipelines, idea is to avoid any extra overhead especially for user-facing scenarios. To that end models with really small footprint that doesn't need any specialised hardware and yet offer competitive performance are chosen. Feel free to raise issues to add support for a new models as you see fit.


## Installation:
```python 
pip install flashrank
```

## Getting started:
```python
from flashrank import Ranker, RerankRequest

# Nano (~4MB), blazing fast model & competitive performance (ranking precision).
ranker = Ranker()

or 

# Small (~34MB), slightly slower & best performance (ranking precision).
ranker = Ranker(model_name="ms-marco-MiniLM-L-12-v2", cache_dir="/opt")

or 

# Medium (~110MB), slower model with best zeroshot performance (ranking precision) on out of domain data.
ranker = Ranker(model_name="rank-T5-flan", cache_dir="/opt")

or 

# Medium (~150MB), slower model with competitive performance (ranking precision) for 100+ languages  (don't use for english)
ranker = Ranker(model_name="ms-marco-MultiBERT-L-12", cache_dir="/opt")

or 

ranker = Ranker(model_name="rank_zephyr_7b_v1_full", max_length=1024) # adjust max_length based on your passage length
```

```python
# Metadata is optional, Id can be your DB ids from your retrieval stage or simple numeric indices.
query = "How to speedup LLMs?"
passages = [
   {
      "id":1,
      "text":"Introduce *lookahead decoding*: - a parallel decoding algo to accelerate LLM inference - w/o the need for a draft model or a data store - linearly decreases # decoding steps relative to log(FLOPs) used per decoding step.",
      "meta": {"additional": "info1"}
   },
   {
      "id":2,
      "text":"LLM inference efficiency will be one of the most crucial topics for both industry and academia, simply because the more efficient you are, the more $$$ you will save. vllm project is a must-read for this direction, and now they have just released the paper",
      "meta": {"additional": "info2"}
   },
   {
      "id":3,
      "text":"There are many ways to increase LLM inference throughput (tokens/second) and decrease memory footprint, sometimes at the same time. Here are a few methods I’ve found effective when working with Llama 2. These methods are all well-integrated with Hugging Face. This list is far from exhaustive; some of these techniques can be used in combination with each other and there are plenty of others to try. - Bettertransformer (Optimum Library): Simply call `model.to_bettertransformer()` on your Hugging Face model for a modest improvement in tokens per second. - Fp4 Mixed-Precision (Bitsandbytes): Requires minimal configuration and dramatically reduces the model's memory footprint. - AutoGPTQ: Time-consuming but leads to a much smaller model and faster inference. The quantization is a one-time cost that pays off in the long run.",
      "meta": {"additional": "info3"}

   },
   {
      "id":4,
      "text":"Ever want to make your LLM inference go brrrrr but got stuck at implementing speculative decoding and finding the suitable draft model? No more pain! Thrilled to unveil Medusa, a simple framework that removes the annoying draft model while getting 2x speedup.",
      "meta": {"additional": "info4"}
   },
   {
      "id":5,
      "text":"vLLM is a fast and easy-to-use library for LLM inference and serving. vLLM is fast with: State-of-the-art serving throughput Efficient management of attention key and value memory with PagedAttention Continuous batching of incoming requests Optimized CUDA kernels",
      "meta": {"additional": "info5"}
   }
]

rerankrequest = RerankRequest(query=query, passages=passages)
results = ranker.rerank(rerankrequest)
print(results)
```

```python 
# Reranked output from default reranker
[
   {
      "id":4,
      "text":"Ever want to make your LLM inference go brrrrr but got stuck at implementing speculative decoding and finding the suitable draft model? No more pain! Thrilled to unveil Medusa, a simple framework that removes the annoying draft model while getting 2x speedup.",
      "meta":{
         "additional":"info4"
      },
      "score":0.016847236
   },
   {
      "id":5,
      "text":"vLLM is a fast and easy-to-use library for LLM inference and serving. vLLM is fast with: State-of-the-art serving throughput Efficient management of attention key and value memory with PagedAttention Continuous batching of incoming requests Optimized CUDA kernels",
      "meta":{
         "additional":"info5"
      },
      "score":0.011563735
   },
   {
      "id":3,
      "text":"There are many ways to increase LLM inference throughput (tokens/second) and decrease memory footprint, sometimes at the same time. Here are a few methods I’ve found effective when working with Llama 2. These methods are all well-integrated with Hugging Face. This list is far from exhaustive; some of these techniques can be used in combination with each other and there are plenty of others to try. - Bettertransformer (Optimum Library): Simply call `model.to_bettertransformer()` on your Hugging Face model for a modest improvement in tokens per second. - Fp4 Mixed-Precision (Bitsandbytes): Requires minimal configuration and dramatically reduces the model's memory footprint. - AutoGPTQ: Time-consuming but leads to a much smaller model and faster inference. The quantization is a one-time cost that pays off in the long run.",
      "meta":{
         "additional":"info3"
      },
      "score":0.00081340264
   },
   {
      "id":1,
      "text":"Introduce *lookahead decoding*: - a parallel decoding algo to accelerate LLM inference - w/o the need for a draft model or a data store - linearly decreases # decoding steps relative to log(FLOPs) used per decoding step.",
      "meta":{
         "additional":"info1"
      },
      "score":0.00063596206
   },
   {
      "id":2,
      "text":"LLM inference efficiency will be one of the most crucial topics for both industry and academia, simply because the more efficient you are, the more $$$ you will save. vllm project is a must-read for this direction, and now they have just released the paper",
      "meta":{
         "additional":"info2"
      },
      "score":0.00024851
   }
]
```

## You can use it with any search & retrieval pipeline:

1. **Lexical Search (RegularDBs that supports full-text search or Inverted Index)**
  <center><img src="./images/lexical_search.png" width=600/></center>

<br/>

2. **Semantic Search / RAG usecases (VectorDBs)**
  <center><img src="./images/vector_search_rag.png" width=600/></center>
<br/>

3. **Hybrid Search**
  <center><img src="./images/hybrid_search.png" width=400/></center>

<br/>

## Deployment patterns
#### How to use it in a AWS Lambda function ?
In AWS or other serverless environments the entire VM is read-only you might have to create your 
own custom dir. You can do so in your Dockerfile and use it for loading the models (and eventually as a cache between warm calls). You can do it during init with cache_dir parameter. 

```python
ranker = Ranker(model_name="ms-marco-MiniLM-L-12-v2", cache_dir="/opt")
```

## References:

1. **In-domain and Zeroshot performance of Cross Encoders fine-tuned on MS-MARCO**
  <center><img src="./images/CE_BEIR.png" width=600/></center>

<br/>

2. **In-domain and Zeroshot performance of RankT5 fine-tuned on MS-MARCO**
  <center><img src="./images/RankT5_BEIR.png" width=450/></center>
<br/>

## How to Cite?

To cite this repository in your work please click the "cite this repository" link on the right side (bewlow repo descriptions and tags)


## Papers citing flashrank

- [COS-Mix: Cosine Similarity and Distance Fusion for
Improved Information Retrieval](https://arxiv.org/pdf/2406.00638)

- [Bryndza at ClimateActivism 2024: Stance, Target and Hate Event
Detection via Retrieval-Augmented GPT-4 and LLaMA](https://arxiv.org/pdf/2402.06549)

- [Stance and Hate Event Detection in Tweets Related to
Climate Activism - Shared Task at CASE 2024](https://aclanthology.org/2024.case-1.33.pdf)
