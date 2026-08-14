# Passage Re-ranking with BERT

A reimplementation of the BERT passage re-ranking approach proposed by Nogueira & Cho.

The project explores how a BERT-based cross-encoder can be used to score query-passage pairs and improve passage ranking after an initial retrieval stage.

This implementation was developed as a reproduction-oriented study while making several engineering and implementation choices to accommodate limited computational resources.

# Original Paper

Nogueira, R., & Cho, K. (2019).

The original work applies BERT to re-rank passages retrieved by an initial retrieval system such as BM25.

# Pipeline

Query
  ↓
Initial Retrieval
  ↓
Candidate Passages
  ↓
Query–Passage Pairs
  ↓
BERT Cross-Encoder
  ↓
Relevance Score
  ↓
Re-ranked Passages
  ↓
MRR@10

# Dataset

This implementation uses the MS MARCO dataset, version 1.1, loaded through the Hugging Face Datasets library.

The dataset contains natural-language queries paired with candidate passages and relevance labels.

Each query-passage pair is represented as:

- `query`
- `query_id`
- `passage`
- `label`
  
# Our Implementation

The implementation follows the core idea of the original work:

 represent a query and a passage as a single BERT input and use BERT to predict passage relevance.

The main implementation choices include:

- Lazy tokenization to reduce memory consumption.
- Query-passage pair construction using a generator-oriented workflow.
- PyTorch Dataset and DataLoader.
- BERT sequence classification architecture.
- AdamW optimizer.
- Linear learning-rate scheduling with warmup.
- Gradient accumulation to achieve an effective batch size of 128 under limited GPU memory.
- MRR@10 as the primary ranking evaluation metric.

# Differences from the Original Paper

This project should not be interpreted as an exact reproduction of the original experimental environment.

Several implementation differences were introduced because the experiments were conducted under limited computational resources.

| Component | Original Work | Our Implementation |
|---|---|---|
| BERT model | BERT-Large | BERT-Base |
| Optimizer | Adam | AdamW |
| Tokenization | Original implementation | Lazy tokenization |
| Batch size | 128 | 128 effective batch |
| Gradient accumulation | Not used in the same way | Used |
| Training steps | 100,000 | Reduced |
| Warmup steps | 10,000 | Reduced |
| Maximum sequence length | 512 | 512 |
| Maximum query length | 64 | 64 |
| Evaluation metric | MRR@10 | MRR@10 |

# Results

## Validation MRR@10

| Experiment | MRR@10 |
|---|---:|
| Original Paper — BERT-Base | 0.347 |
| Original Paper — BERT-Large | 0.365 |
| Our Implementation | 0.4603 |

**Important:** These values should not be interpreted as a direct apples-to-apples comparison. Our implementation differs from the original experimental setup in model size, training budget, dataset processing, and other implementation details. The comparison is provided as a reference point rather than a claim of superiority.

# Experimental Environment

| Property | Value |
|---|---|
| GPU | NVIDIA GeForce RTX 5050 Laptop GPU |
| Peak GPU Memory | ~2.98 GB |
| Parameters | ~109.5M |
| Training Time | ~2.17 hours |
| Effective Batch Size | 128 |
| Maximum Sequence Length | 512 |

# Future Work

Future research will investigate:

- Hard-negative sampling strategies.
- BM25 vs dense vs mixed negatives.
- Computation-aware re-ranking.
- Adaptive reranking depth.
- Effectiveness–efficiency trade-offs.
- Cross-domain evaluation.
