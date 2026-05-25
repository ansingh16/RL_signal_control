# Traffic Congestion Prediction with LSTM on TfL Data

Binary classification of traffic congestion on the M25 motorway using an LSTM neural network trained on real Transport for London (TfL) sensor data. The model ingests sequences of traffic volume, speed, and temporal features recorded at 15-minute intervals across dozens of M25 checkpoints and predicts whether the next interval will be congested (average speed below 30 mph).

## Data Pipeline

```
TfL API (pytris)  -->  raw parquet files (per checkpoint)  -->  feature engineering  -->  LSTM classifier
```

1. **Fetch** -- `notebooks/data_fetch.py` queries the TfL daily-reports API via the `pytris` library for up to 3 years of 15-minute traffic data across ~90 M25 checkpoint sites. Requests run in parallel with exponential-backoff retry logic.
2. **Store** -- Each site is saved as an individual parquet file under `data/raw/`.
3. **Feature engineering** -- `TrafficDataProcessor` in `scripts/Traffic.py` loads a parquet file and produces:
   - Core features: Total Volume, Avg mph
   - Lag features at 15 min, 30 min, and 1 h offsets (volume and speed)
   - Rolling statistics (mean, std) over 1 h, 2 h, and 4 h windows
   - Time features: hour of day, day of week
   - Target label: `congestion = 1` when Avg mph < 30
4. **Sequence creation** -- `TrafficDataset` forms sliding-window sequences (default length 16 intervals = 4 hours) for the LSTM.
5. **Splitting** -- Chronological train/val/test split (default 70/15/15) to prevent temporal leakage. Optional `WeightedRandomSampler` handles class imbalance.

## Model Architecture

`TrafficNet` in `scripts/Traffic.py`:

```
Input (seq_len x num_features)
  |
LSTM (configurable hidden_size, num_layers, dropout)
  |
Last hidden state
  |
Linear  -->  ReLU  -->  Dropout  -->  Linear  -->  2-class logits (CrossEntropyLoss)
```

Default configuration: hidden_size=64, num_layers=2, dropout=0.2. A conservative variant (hidden_size=16, num_layers=1, dropout=0.5) is available in `scripts/diagnosis.py` for regularisation experiments.

Training uses Adam with weight decay, ReduceLROnPlateau scheduling, optional weighted cross-entropy loss for class imbalance, and early stopping on validation loss.

## Baseline Comparisons

`scripts/diagnosis.py` provides three baselines to benchmark the LSTM against:

| Baseline | Method |
|---|---|
| Majority class | Always predict the most frequent label |
| Rush-hour rules | Predict congestion during weekday rush hours (7--9 AM, 5--7 PM) |
| Speed threshold | Predict congestion when Avg mph < 25 |

A full diagnostic pipeline (`run_diagnostics_and_retrain`) also checks feature-target correlations, class distribution over time, and temporal congestion patterns.

## Project Structure

```
.
|-- scripts/
|   |-- Traffic.py          # TrafficDataset, TrafficDataProcessor, TrafficNet, ModelTrainer, ResultsPlotter
|   |-- diagnosis.py        # Data-quality diagnostics, baselines, conservative model, improved training setup
|-- notebooks/
|   |-- data_fetch.py       # Parallel TfL API data collection via pytris
|   |-- Data_exploration.ipynb  # Exploratory data analysis
|-- data/
|   |-- raw/                # ~90 parquet files (one per M25 checkpoint) + check_points.csv
|   |-- processed/          # dft_traffic_A406.csv
|-- Plots/
|   |-- TrainingHistory.png # Training loss, accuracy, and F1 curves
|-- models/                 # Saved model checkpoints (gitignored)
|-- .env                    # TfL API credentials (gitignored)
|-- .gitignore
|-- LICENSE
|-- requirements.txt
```

## How to Run

### Prerequisites

```bash
pip install -r requirements.txt
```

### 1. Get a TfL API key

Register at [https://api-portal.tfl.gov.uk/](https://api-portal.tfl.gov.uk/) and add your credentials to a `.env` file in the project root.

### 2. Fetch traffic data

```bash
cd notebooks
python data_fetch.py
```

This downloads 3 years of 15-minute traffic data for M25 checkpoints into `data/raw/`. The script skips sites that have already been fetched.

### 3. Train the model

From a Python script or notebook:

```python
from scripts.Traffic import TrafficDataProcessor, TrafficNet, ModelTrainer

processor = TrafficDataProcessor("data/raw/df_10268.parquet")
processor.prepare_site_data(speed_threshold=30)

train_loader, val_loader, test_loader, info = processor.create_loaders_from_df(
    seq_len=16, batch_size=64, handle_imbalance=True
)

model = TrafficNet(input_size=len(info["feature_cols"]))
trainer = ModelTrainer(model)
trained_model = trainer.train_model(train_loader, val_loader, epochs=50, early_stopping_patience=10)
```

### 4. Run diagnostics and baselines

```python
from scripts.diagnosis import run_diagnostics_and_retrain

trainer, train_loader, val_loader, test_loader, baselines = run_diagnostics_and_retrain(processor, seq_len=48)
```

## Tech Stack

- **PyTorch** -- LSTM model, training loop, data loading
- **pytris** -- Python client for the TfL API
- **pandas / NumPy** -- Data processing and feature engineering
- **scikit-learn** -- Evaluation metrics (accuracy, precision, recall, F1, confusion matrix)
- **plotly / matplotlib** -- Visualisation and training curves
- **tqdm** -- Progress bars for data fetching
- **python-dotenv** -- Environment variable management for API credentials
