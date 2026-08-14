# Passage Re-ranking with BERT

A reimplementation of the BERT passage re-ranking approach proposed by Nogueira & Cho.

The project explores how a BERT-based cross-encoder can be used to score query-passage pairs and improve passage ranking after an initial retrieval stage.

This implementation was developed as a reproduction-oriented study while making several engineering and implementation choices to accommodate limited computational resources.

# Original Paper

Nogueira, R., & Cho, K. (2019).

The original work applies BERT to re-rank passages retrieved by an initial retrieval system such as BM25.

# Pipeline

```text
Query
  ↓
Candidate Passages
  ↓
Query–Passage Pairs
  ↓
BERT Cross-Encoder
  ↓
Relevance Score
  ↓
Ranking
  ↓
MRR@10
'''

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

# Lazy Tokenization

Instead of tokenizing the complete dataset and storing all encoded sequences in memory, tokenization is performed when samples are requested by the PyTorch Dataset.

This reduces memory consumption and makes the implementation more suitable for limited GPU/CPU environments.

# Effective Batch Size

A micro-batch is used because of GPU memory constraints.

Micro Batch = 4

Gradient Accumulation = 32

Effective Batch Size = 128

# Experimental Configuration

| Parameter               | Value                        |
| ----------------------- | ---------------------------- |
| Model                   | BERT-Base                    |
| Dataset                 | MS MARCO v1.1                |
| Optimizer               | AdamW                        |
| Learning Rate           | 3e-6                         |
| Weight Decay            | 0.01                         |
| Scheduler               | Linear Warmup + Linear Decay |
| Warmup Steps            | 100                          |
| Training Steps          | 1,000                        |
| Effective Batch Size    | 128                          |
| Micro Batch Size        | 4                            |
| Gradient Accumulation   | 32                           |
| Maximum Sequence Length | 512                          |
| Maximum Query Length    | 64                           |
| Dropout                 | 0.1                          |
| Seed                    | 42                           |
| Evaluation Metric       | MRR@10                       |

# Experimental Environment

| Property                | Value                              |
| ----------------------- | ---------------------------------- |
| GPU                     | NVIDIA GeForce RTX 5050 Laptop GPU |
| Peak GPU Memory         | ~2.98 GB                           |
| Parameters              | ~109.5M                            |
| Training Time           | ~2.17 hours                        |
| Effective Batch Size    | 128                                |
| Maximum Sequence Length | 512                                |


# Original Paper vs Our Implementation

| Component       | Original Work              | Our Implementation  |
| --------------- | -------------------------- | ------------------- |
| BERT            | BERT-Large                 | BERT-Base           |
| Optimizer       | Adam                       | AdamW               |
| Tokenization    | Original implementation    | Lazy tokenization   |
| Batch Size      | 128                        | 128 effective       |
| Training Budget | 100k iterations            | 1k optimizer steps  |
| Warmup          | 10k steps                  | 100 steps           |
| Sequence Length | 512                        | 512                 |
| Query Length    | 64                         | 64                  |
| Evaluation      | MRR@10                     | MRR@10              |
| Hardware        | Original paper environment | RTX 5050 Laptop GPU |

# Results
The Notebook produce:
Training Loss curve
Validation MRR@10
Paper vs Implementation comparison
Experiment Configuration
Results Dashboard

**Important: ** These values should not be interpreted as a direct apples-to-apples comparison. Our implementation differs from the original experimental setup in model size, training budget, dataset processing, and other implementation details. The comparison is provided as a reference point rather than a claim of superiority.

# Limitations

This implementation is not an exact reproduction of the original experimental environment.

The main limitations are:

BERT-Base is used instead of BERT-Large.
The training budget is substantially smaller.
The warmup schedule is reduced.
AdamW is used instead of the original optimization setup.
Lazy tokenization is used for memory efficiency.
Gradient accumulation is used to obtain an effective batch size of 128.
The complete original BM25 Top-1000 candidate-generation protocol is not reproduced in the current implementation.

Consequently, the reported MRR@10 should be interpreted as an implementation result rather than a direct replication result.

# Repository Structure

passage-reranking-with-bert/
│
├── README.md
├── requirements.txt
├── bert_passage_reranking.ipynb
│
└── results/
    ├── training_loss.png
    ├── validation_mrr.png
    ├── paper_comparison.png
    ├── experiment_configuration.csv
    └── results.json


# Future Research

The current repository establishes the implementation baseline for future research into efficient neural passage re-ranking.

Potential directions include:

Hard-negative sampling
Random vs BM25 vs dense negatives
Mixed hard-negative strategies
Hard-negative-aware BERT training
Computation-aware re-ranking
Adaptive reranking depth
Effectiveness–efficiency evaluation
Cross-domain validation
Ablation studies
Statistical significance testing
Failure analysis

These directions are intentionally kept separate from the current reimplementation.

# Citation

If you use this repository, please cite the original work:
@article{nogueira2019passage,
  title={Passage Re-ranking with BERT},
  author={Nogueira, Rodrigo and Cho, Kyunghyun},
  journal={arXiv preprint arXiv:1901.04085},
  year={2019}
}

# License

This project is intended for educational and research purposes.
