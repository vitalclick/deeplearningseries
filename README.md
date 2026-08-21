# Deep Learning Series

A collection of Jupyter notebooks from a deep learning / machine learning video
series. Each notebook is a self-contained walkthrough of one problem, taken from
raw data all the way through to a trained model and an evaluation of how well it
did.

The notebooks were written for Azure Notebooks / Azure Machine Learning
Workbench, so a few of them use Microsoft-specific pieces (CNTK as a backend,
the `azureml` data-prep package). The general workflow in each is standard
Python data science and translates to any environment.

## Notebooks

| Notebook | Problem | Approach | Stack |
| --- | --- | --- | --- |
| [`Wine_Quality.ipynb`](Wine_Quality.ipynb) | Predict the quality rating of red wine from its physicochemical properties | Exploratory analysis (quality distribution, correlation heatmap), feature scaling, multi-class logistic regression | pandas, seaborn, matplotlib, scikit-learn |
| [`ImageClassifier.ipynb`](ImageClassifier.ipynb) | Recognise handwritten digits (MNIST) | Feed-forward network — one 200-unit ReLU hidden layer, softmax over 10 classes, trained with AdaDelta for 10 sweeps | CNTK, NumPy, SciPy |
| [`flight_data_prepare.ipynb`](flight_data_prepare.ipynb) | Predict whether a flight will arrive more than 10 minutes late | One-hot encoding of carrier/origin/destination/hour features, then a dense 512-unit sigmoid network with dropout and a softmax output | Azure ML data prep, Keras, pandas, scikit-learn |
| [`stock_price_predictor.ipynb`](stock_price_predictor.ipynb) | Forecast the next closing price of MSFT from the previous 7 days | Min-max normalisation, sliding-window time series, single-layer LSTM regressor trained for 100 epochs | Keras (CNTK backend), pandas, scikit-learn |

## Results recorded in the notebooks

These are the numbers stored in the committed cell outputs — they come from a
single run with no hyperparameter tuning, so treat them as a starting point
rather than a benchmark.

- **Wine quality** — 0.575 accuracy on a 20% held-out test set (10-ish quality
  classes, unbalanced).
- **MNIST** — 0.0232 classification error (~97.7% accuracy) on the 10,000-image
  test set.
- **Flight delays** — 0.848 accuracy on the held-out 20%. The confusion counts
  (TP 238, FP 0, TN 1417, FN 297) show the model is heavily biased towards
  predicting "on time", which is worth keeping in mind before reading much into
  the headline accuracy.
- **Stock price** — RMSE 0.72 on train, 4.94 on test, i.e. it fits the training
  window far better than it generalises.

## Data

None of the datasets are checked in. Each notebook expects its data to be
available locally:

- **MNIST** is downloaded by the notebook itself from Yann LeCun's site and
  converted to the CNTK text format (`Train-28x28_cntk_text.txt`,
  `Test-28x28_cntk_text.txt`). The final cell also reads a hand-drawn `3.png` to
  test the model on your own digit.
- **Wine quality** expects `winequality-red.csv` (semicolon-separated) in the
  working directory — the red-wine file from the UCI Machine Learning
  Repository's Wine Quality dataset.
- **Flight delays** loads `flight_data_prepare.dprep`, an Azure ML Workbench
  data-preparation package built from US DOT on-time performance data. The
  `.dprep` file is not in the repo, so this notebook only runs inside a Workbench
  project.
- **Stock prices** expects `~/library/HistoricalQuotes.csv`, an export of MSFT
  historical quotes with `date`, `close`, `volume`, `open`, `high` and `low`
  columns.

## Running the notebooks

The notebooks were written against Python 3.5/3.6 and the library versions of
the time; some calls have since been removed from their libraries (for example
`sklearn.cross_validation` and `scipy.ndimage.imread`), so expect to adjust a
line or two on a modern stack.

```bash
python -m venv .venv
source .venv/bin/activate
pip install jupyter numpy pandas matplotlib seaborn scikit-learn scipy keras
jupyter notebook
```

Additional per-notebook requirements:

- `ImageClassifier.ipynb` needs [CNTK](https://learn.microsoft.com/en-us/cognitive-toolkit/).
- `stock_price_predictor.ipynb` sets the Keras backend to CNTK in its first
  cell; drop that cell to run it on TensorFlow instead.
- `flight_data_prepare.ipynb` needs the `azureml` data-prep and logging packages
  and the accompanying Workbench project.

## Repository layout

```
.
├── ImageClassifier.ipynb        # MNIST digit classification with CNTK
├── Wine_Quality.ipynb           # Red wine quality with logistic regression
├── flight_data_prepare.ipynb    # Flight delay prediction with Keras
└── stock_price_predictor.ipynb  # MSFT close price with an LSTM
```
