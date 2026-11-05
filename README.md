# AML GNN for SAML-D

A PyTorch project for detecting suspicious transfers in the synthetic SAML-D dataset.

The model classifies individual transactions using their attributes and the historical transaction graph of the sender, receiver, and their counterparties.

Data (SAML-D): https://github.com/BOztasUK/Anti_Money_Laundering_Transaction_Data_SAML-D

## Model overview

- **Nodes:** bank accounts.
- **Directed edges:** transactions between accounts.
- **Target:** the transaction-level `Is_laundering` label.
- **Transaction features:** amount, time, currencies, bank locations, and payment type.
- **Graph context:** historical incoming and outgoing transactions.
- **Output:** a risk score and an alert based on a validation-selected threshold.

The model only uses transactions strictly earlier than the transaction
being scored. Transactions with identical timestamps cannot use one
another as historical context.

`Laundering_type` and transaction labels are excluded from model inputs.

Message passing is implemented directly in PyTorch.
**PyTorch Geometric is not required.**

## Project Structure

```text
aml_gnn/
├── prepare.py       # Validate CSV, build features, splits, and graph indices
├── graph.py         # Sample historical neighborhoods and construct batches
├── model.py         # Encode transactions and perform graph message passing
├── run.py           # Train, evaluate, and score transactions
├── data/            # Input CSV files, downloaded separately
└── artifacts/       # Processed arrays, checkpoints, and predictions

`data/` and `artifacts/` are suggested directory names, not framework
requirements. Paths are supplied through command-line arguments.

Create `data/` and place your CSV files there. Output directories are
created automatically by the scripts.

Run the following commands from the project root.
Python 3.12 or higher is recommended.

## installation


Create a virtual environment:

```bash
python -m venv .venv
```

Activate it on Linux or macOS:

```bash
source .venv/bin/activate
```

Activate it on Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

Install dependencies:

```bash
python -m pip install torch numpy pandas scikit-learn
```

For GPU execution, install a PyTorch build compatible with your system:
https://pytorch.org/get-started/locally/

Record the installed package versions:

```bash
python -m pip freeze > requirements-lock.txt
```

## Dataset

Download SAML-D using the dataset link in the authors' repository:
https://github.com/BOztasUK/Anti_Money_Laundering_Transaction_Data_SAML-D

Place the downloaded CSV at:

```text
data/SAML-D.csv
```

The required input columns are:

```text
Date
Time
Amount
Sender_account
Receiver_account
Payment_currency
Received_currency
Sender_bank_location
Receiver_bank_location
Payment_type
```

Training additionally requires:

```text
Is_laundering
```

Labels must be `0` or `1`.

The default combined date/time format is:

```text
%Y-%m-%d %H:%M:%S
```

Use `prepare.py --time-format` if your dataset uses a different format.

## Training and evaluation

### 1. Prepare the Data

```bash
python prepare.py --csv data/SAML-D.csv --out artifacts/dataset
```

The preparation script:

1. Validates required columns and values.
2. Parses timestamps and sorts transactions chronologically.
3. Splits the observed days into approximately 70% training, 15% validation, and 15% test.
4. Fits categorical vocabularies and numeric normalization on training data only.
5. Converts account identifiers into graph node indices.
6. Builds incoming and outgoing transaction indices.
7. Saves processed arrays and preprocessing metadata.

Generated files include:

```text
artifacts/dataset/
├── time.npy
├── src.npy
├── dst.npy
├── num.npy
├── cat.npy
├── y.npy
├── split.npy
├── row_id.npy
├── in_order.npy
├── in_ptr.npy
├── in_time.npy
├── out_order.npy
├── out_ptr.npy
├── out_time.npy
├── accounts.csv
├── preprocess.json
└── manifest.json
```

Check `manifest.json` for row counts and positive examples in each split.

The preparation output directory must be empty or nonexistent.

### 2. Обучение

```bash
python run.py train --data artifacts/dataset --out artifacts/gnn --epochs 10 --batch-size 256 --depth 2 --fanout 4 --alert-rate 0.001
```

#### Main Options

| Option | Description |
|---|---|
| `--epochs` | Maximum number of training epochs |
| `--batch-size` | Number of target transactions per batch |
| `--depth` | Number of message-passing levels: `0`, `1`, or `2` |
| `--fanout` | Maximum historical neighbors per direction and level |
| `--hidden` | Hidden representation size |
| `--lr` | Learning rate |
| `--patience` | Early-stopping patience |
| `--pos-weight` | Positive-class loss weight |
| `--alert-rate` | Validation review fraction used to select a threshold |
| `--device` | Execution device, such as `cpu` or `cuda` |
| `--seed` | Random seed |

By default, the positive-class weight is calculated as the ratio of
negative to positive training examples.

The best model is selected using validation Average Precision (AP).

The training output directory must be empty or nonexistent.

#### Saved Artifacts

```text
artifacts/gnn/
├── model.pt
├── best_unfinalized.pt
├── preprocess.json
└── validation_metrics.json
```

`model.pt` contains:

- Model weights.
- Architecture configuration.
- Neighborhood sampling settings.
- Preprocessing configuration.
- Selected alert threshold.
- Training metadata.

Use `model.pt` for evaluation and scoring.

`best_unfinalized.pt` is an intermediate checkpoint without the finalized
alert threshold. It is not intended for normal scoring.

### 3. Evaluate on the Test Period

```bash
python run.py evaluate --data artifacts/dataset --checkpoint artifacts/gnn/model.pt --out artifacts/test
```

Outputs:

```text
artifacts/test/
├── test_metrics.json
├── predictions.csv
└── pr_curve.npz
```

Reported metrics include:

- Average Precision.
- ROC-AUC.
- Precision, recall, and F1 at the saved threshold.
- Alert count.
- False positives and false negatives.
- Precision and recall within a daily review budget.

Use validation data to choose hyperparameters and thresholds.
Reserve test data for the final evaluation.

The daily top-K metrics represent ranking after all transactions for
a day are available. They are not equivalent to online thresholding.

## 4. Score New Transactions

Prepare two input files:

- `data/history.csv`: previously observed transactions.
- `data/new_transactions.csv`: a subsequent batch to score.

Labels are not required and are not read in scoring mode.

Prepare the batch using the saved preprocessing configuration:

```bash
python prepare.py --csv data/new_transactions.csv --history data/history.csv --encoder artifacts/gnn/preprocess.json --out artifacts/new_batch
```

Run scoring:

```bash
python run.py score --data artifacts/new_batch --checkpoint artifacts/gnn/model.pt --out artifacts/new_predictions
```

Predictions are saved to:

```text
artifacts/new_predictions/predictions.csv
```

## Model Architecture

The network consists of:

1. A transaction encoder for numeric and categorical features.
2. An account encoder for historical activity statistics.
3. Message-passing layers using counterparty representations,
   historical transaction features, and transaction age.
4. Separate aggregation of incoming and outgoing messages.
5. A transaction classification head combining sender, receiver,
   and current transaction representations.

Account identifiers are used only to address the graph.
There are no trainable account-ID embeddings.

With `--depth 1` or `--depth 2`, the model uses graph message passing.

With `--depth 0`, it uses transaction attributes and causal account
statistics without messages from counterparties.

## Check Whether the Graph Helps

Train a control model without message passing:

```bash
python run.py train --data artifacts/dataset --out artifacts/no_graph --depth 0 --epochs 10 --alert-rate 0.001
```

Compare validation metrics against models with `--depth 1` and
`--depth 2`.

Graph structure should provide a measurable improvement; using a GNN
alone does not establish that graph information is useful.

## Limitations

- CSV preparation currently loads and sorts the input in RAM.
  The full dataset requires substantial memory.
- Training uses mini-batches and memory-mapped arrays, but neighborhood
  size grows with depth and fanout.
- Only the most recent sampled transactions are used for message passing.
  Older events and longer laundering chains may be missed.
- Historical events must be available at scoring time.
- All transaction attributes must be known when a prediction is made.
  Pre-execution scoring requires a separate feature-availability review.
- `risk_score` is not a calibrated probability.
- Strong performance on synthetic data does not demonstrate equivalent
  performance on real banking data.
- The project implements batch scoring, not a streaming service.
  Incremental graph storage and late-event handling are not included.
- Training results, runtime, and detection quality must be measured on
  the downloaded dataset version; no benchmark results are supplied.
