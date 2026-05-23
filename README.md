## MOEX Kronos Rolling Regime Selector

This fork contains a reproducible research pipeline for a MOEX Top-20
intraday portfolio selector built on top of Kronos forecasts, daily/news
background scores, ALGOPACK execution costs, and live-safe rolling selectors.

The best currently checked live-safe setup is **`rolling_best_w24`**:

- every 30-minute decision point chooses between `family_first`,
  `news_aware`, and `marketwide_news`;
- the choice uses only each selector's realized return over the previous
  24 intervals, never future returns;
- the underlying scenario grid uses Kronos rank, LLM/news rank, threshold
  and hyperbolic allocation variants;
- execution costs are already included through ALGOPACK `OBStats` BBO/depth
  plus `0.03%` entry and exit commission.

Reference local research results, all starting from 100,000 RUB per window:

| Evaluation window | Selector | Final capital | Return |
| --- | --- | ---: | ---: |
| 2026-04-01 .. 2026-04-14 | `rolling_best_w24` | 113,701.57 RUB | +13.70% |
| 2026-04-14 .. 2026-04-30 | `rolling_best_w24` | 110,913.45 RUB | +10.91% |
| 2026-05-01 .. 2026-05-21 | `rolling_best_w24` | 120,755.49 RUB | +20.76% |
| all three windows compounded | `rolling_best_w24` | 152,285.14 RUB | +52.29% |

For comparison, the same three-window compounded returns were:

| Selector | Compounded return | Avg window return |
| --- | ---: | ---: |
| `rolling_best_w24` | +52.29% | +15.12% |
| `rolling_rank_weighted_w24_p2` | +50.79% | +14.75% |
| `rolling_best_ff_na_w24` without market-wide news | +52.05% | +15.07% |
| `selector_no_news` | +49.85% | +14.50% |
| `selector_family_first` | +49.61% | +14.46% |
| `selector_marketwide_news` | +48.49% | +14.16% |
| `selector_news_aware` | +47.09% | +13.81% |
| `baseline_lightgbm_legacy` | +33.00% | +9.97% |

The oracle rows produced during research are **not live-safe**. They are kept
only as upper bounds for diagnostics.

`rolling_rank_weighted_w24_p2` is the softer variant: it ranks
`family_first`, `news_aware`, and `marketwide_news` by the previous 24
intervals, then allocates all three with rank weights `rank^-2`. It was more
diversified than `rolling_best_w24`, but the hard rolling pick still performed
best in the checked windows.

### What the pipeline does

1. Builds Kronos MC10 forecasts for the fixed Top-20 MOEX universe.
2. Scores each rebalance timestamp with Polza AI news-background sentiment.
3. Builds 7,000 hyperbolic portfolio scenarios from Kronos rank and LLM rank.
4. Reprices execution with MOEX ALGOPACK historical `OBStats`.
5. Trains rolling LightGBM selectors and diagnostic ablations using only past
   intervals.
6. Combines the strongest live-safe selector families through
   `rolling_best_w24`.

Top-20 universe:

```text
LKOH SBER ROSN GAZP VTBR YDEX PLZL T NVTK X5
GMKN MGNT ALRS AFLT CHMF NLMK MOEX SNGSP MTSS PIKK
```

### Required keys

Set the keys in your local shell or profile. Do not write real keys into
tracked files.

PowerShell example:

```powershell
$env:POLZA_AI_API_KEY = "<your Polza AI key>"
$env:MOEX_ALGO_TOKEN = "<your MOEX ALGOPACK token>"
$env:KRONOS_WEIGHTS_DIR = "D:\zxcza"
```

`POLZA_AI_API_KEY` is used for news-background scoring through
`https://polza.ai/api/v1`. `MOEX_ALGO_TOKEN` is used for ALGOPACK `OBStats`
downloads. You can also provide a local token file through runner arguments,
but do not commit it.

### Install

Use a CPU Python/conda environment for the selector. LightGBM and CatBoost are
not required inside the CUDA Kronos environment.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

For Kronos inference on CUDA you can use a separate CUDA environment; for the
LightGBM selector itself, normal CPU Python is enough.

### Reproduce the selector from existing compact data

The latest combined selector can be reproduced from existing selector outputs:

```powershell
python run_moex_top20_selector_v2_combiner.py `
  --out-dir outputs_moex_kronos_top20_selector_v2_combined_family_news
```

The combiner expects these selector output folders to exist by default:

- `outputs_moex_kronos_top20_selector_v2_jan_may_eval_apr1_14`
- `outputs_moex_kronos_top20_selector_v2_jan_may_eval_apr14_30`
- `outputs_moex_kronos_top20_selector_v2_jan_may_eval_may1_21`

To run the underlying selector v2 ablation for a new evaluation window:

```powershell
python run_moex_top20_selector_v2_lightgbm_ablation.py `
  --compact-dir outputs_moex_kronos_top20_selector_v2_jan_may_eval_may1_21 `
  --out-dir outputs_moex_kronos_top20_selector_v2_custom_eval `
  --eval-from 2026-05-01 `
  --eval-till 2026-05-21 `
  --train-lookback-intervals 720 `
  --max-train-rows 90000 `
  --lightgbm-estimators 80
```

The LightGBM selector expects a grid root containing a `no_anchor` folder with:

- `compact_interval_returns.csv`
- `compact_market_features.csv`
- `compact_scenario_manifest.csv`
- `compact_validation.csv`

Then run:

```powershell
python run_moex_top20_intraday_selector_catboost_lightgbm.py `
  --grid-root outputs_moex_kronos_top20_compact_grid_warm_apr1_may21_rftoday_for_models `
  --sklearn-root outputs_moex_kronos_top20_intraday_selector_may13_21_rftoday_warm_apr_history `
  --out-dir outputs_moex_kronos_top20_catboost_lightgbm_selector_may13_21_rftoday_warm_apr_history `
  --variants no_anchor `
  --models lightgbm `
  --min-meta-train-intervals 96 `
  --meta-retrain-intervals 24 `
  --meta-max-train-rows 80000 `
  --meta-train-lookback-intervals 240 `
  --ensemble-top-n 3 `
  --ensemble-temperature 0.25
```

To rebuild the compact grid, run the news-rank builder with your local
candidate CSVs and news CSV:

```powershell
python run_moex_top20_news_rank_compact_grid_apr_may.py `
  --out-dir outputs_moex_kronos_top20_news_rank_compact_grid_custom `
  --apr-candidates path\to\apr_or_warm_candidates.csv `
  --may-candidates path\to\may_candidates.csv `
  --news-path path\to\news_with_tickers.csv `
  --news-ticker-column tickers `
  --from-date 2026-05-13 `
  --till-date 2026-05-21 `
  --llm-score-mode api
```

Outputs are intentionally ignored by git. They can be regenerated locally with
your data, Polza AI key, and ALGOPACK token.

### Important files

- `configs/best_lightgbm_selector.yaml` - fixed best research configuration.
- `run_moex_top20_news_rank_compact_grid_apr_may.py` - compact grid builder.
- `run_moex_top20_intraday_selector_catboost_lightgbm.py` - live-safe
  LightGBM/CatBoost selector.
- `run_moex_top20_selector_v2_lightgbm_ablation.py` - extended LightGBM
  ablations and diagnostics.
- `run_moex_top20_selector_v2_combiner.py` - combines `family_first`,
  `news_aware`, and `marketwide_news` into `rolling_best_w24`.
- `run_moex_top20_selector_v2_compact_jan_may.py` - builds the Jan-May
  compact selector v2 dataset with explicit missing-news flags.

---

<div align="center">
  <h2><b>Kronos: A Foundation Model for the Language of Financial Markets </b></h2>
</div>


<div align="center">

</a> 
<a href="https://huggingface.co/NeoQuasar"> 
<img src="https://img.shields.io/badge/🤗-Hugging_Face-yellow" alt="Hugging Face"> 
</a> 
<a href="https://shiyu-coder.github.io/Kronos-demo/"> <img src="https://img.shields.io/badge/🚀-Live_Demo-brightgreen" alt="Live Demo"> </a>
<a href="https://github.com/shiyu-coder/Kronos/graphs/commit-activity"> 
<img src="https://img.shields.io/github/last-commit/shiyu-coder/Kronos?color=blue" alt="Last Commit"> 
</a> 
<a href="https://github.com/shiyu-coder/Kronos/stargazers"> 
<img src="https://img.shields.io/github/stars/shiyu-coder/Kronos?color=lightblue" alt="GitHub Stars"> 
</a> 
<a href="https://github.com/shiyu-coder/Kronos/network/members"> 
<img src="https://img.shields.io/github/forks/shiyu-coder/Kronos?color=yellow" alt="GitHub Forks"> 
</a> 
<a href="./LICENSE"> 
<img src="https://img.shields.io/github/license/shiyu-coder/Kronos?color=green" alt="License"> 
</a>

</div>

<div align="center">
  <!-- Keep these links. Translations will automatically update with the README. -->
  <a href="https://zdoc.app/de/shiyu-coder/Kronos">Deutsch</a> | 
  <a href="https://zdoc.app/es/shiyu-coder/Kronos">Español</a> | 
  <a href="https://zdoc.app/fr/shiyu-coder/Kronos">Français</a> | 
  <a href="https://zdoc.app/ja/shiyu-coder/Kronos">日本語</a> | 
  <a href="https://zdoc.app/ko/shiyu-coder/Kronos">한국어</a> | 
  <a href="https://zdoc.app/pt/shiyu-coder/Kronos">Português</a> | 
  <a href="https://zdoc.app/ru/shiyu-coder/Kronos">Русский</a> | 
  <a href="https://zdoc.app/zh/shiyu-coder/Kronos">中文</a>
</div>

<p align="center">

<img src="./figures/logo.png" width="100">

</p>

> Kronos is the **first open-source foundation model** for financial candlesticks (K-lines), 
> trained on data from over **45 global exchanges**.


</div>

## 📰 News
*   🚩 **[2025.11.10]** Kronos has been accpeted by AAAI 2026.
*   🚩 **[2025.08.17]** We have released the scripts for fine-tuning! Check them out to adapt Kronos to your own tasks.
*   🚩 **[2025.08.02]** Our paper is now available on [arXiv](https://arxiv.org/abs/2508.02739)!

<p align="center">

## 📜 Introduction

**Kronos** is a family of decoder-only foundation models, pre-trained specifically for the "language" of financial markets—K-line sequences. Unlike general-purpose TSFMs, Kronos is designed to handle the unique, high-noise characteristics of financial data. It leverages a novel two-stage framework: 
1. A specialized tokenizer first quantizes continuous, multi-dimensional K-line data (OHLCV) into **hierarchical discrete tokens**. 
2. A large, autoregressive Transformer is then pre-trained on these tokens, enabling it to serve as a unified model for diverse quantitative tasks.

<p align="center">
    <img src="figures/overview.png" alt="" align="center" width="700px" />
</p>

## ✨ Live Demo 
We have set up a live demo to visualize Kronos's forecasting results. The webpage showcases a forecast for the **BTC/USDT** trading pair over the next 24 hours. 

**👉 [Access the Live Demo Here](https://shiyu-coder.github.io/Kronos-demo/)** 

## 📦 Model Zoo 
We release a family of pre-trained models with varying capacities to suit different computational and application needs. All models are readily accessible from the Hugging Face Hub.

| Model        | Tokenizer                                                                       | Context length | Params  | Open-source                                                               |
|--------------|---------------------------------------------------------------------------------| -------------- | ------ |---------------------------------------------------------------------------|
| Kronos-mini  | [Kronos-Tokenizer-2k](https://huggingface.co/NeoQuasar/Kronos-Tokenizer-2k)     | 2048           | 4.1M   | ✅ [NeoQuasar/Kronos-mini](https://huggingface.co/NeoQuasar/Kronos-mini)  |
| Kronos-small | [Kronos-Tokenizer-base](https://huggingface.co/NeoQuasar/Kronos-Tokenizer-base) | 512            | 24.7M  | ✅ [NeoQuasar/Kronos-small](https://huggingface.co/NeoQuasar/Kronos-small) |
| Kronos-base  | [Kronos-Tokenizer-base](https://huggingface.co/NeoQuasar/Kronos-Tokenizer-base) | 512            | 102.3M | ✅ [NeoQuasar/Kronos-base](https://huggingface.co/NeoQuasar/Kronos-base)   |
| Kronos-large | [Kronos-Tokenizer-base](https://huggingface.co/NeoQuasar/Kronos-Tokenizer-base) | 512            | 499.2M | ❌                                                                         |


## 🚀 Getting Started

### Installation

1. Install Python 3.10+, and then install the dependencies:

```shell
pip install -r requirements.txt
```

### 📈 Making Forecasts

Forecasting with Kronos is straightforward using the `KronosPredictor` class. It handles data preprocessing, normalization, prediction, and inverse normalization, allowing you to get from raw data to forecasts in just a few lines of code.

**Important Note**: The `max_context` for `Kronos-small` and `Kronos-base` is **512**. This is the maximum sequence length the model can process. For optimal performance, it is recommended that your input data length (i.e., `lookback`) does not exceed this limit. The `KronosPredictor` will automatically handle truncation for longer contexts.

Here is a step-by-step guide to making your first forecast.

#### 1. Load the Tokenizer and Model

First, load a pre-trained Kronos model and its corresponding tokenizer from the Hugging Face Hub.

```python
from model import Kronos, KronosTokenizer, KronosPredictor

# Load from Hugging Face Hub
tokenizer = KronosTokenizer.from_pretrained("NeoQuasar/Kronos-Tokenizer-base")
model = Kronos.from_pretrained("NeoQuasar/Kronos-small")
```

#### 2. Instantiate the Predictor

Create an instance of `KronosPredictor`, passing the model, tokenizer, and desired device.

```python
# Initialize the predictor
predictor = KronosPredictor(model, tokenizer, max_context=512)
```

#### 3. Prepare Input Data

The `predict` method requires three main inputs:
-   `df`: A pandas DataFrame containing the historical K-line data. It must include columns `['open', 'high', 'low', 'close']`. `volume` and `amount` are optional.
-   `x_timestamp`: A pandas Series of timestamps corresponding to the historical data in `df`.
-   `y_timestamp`: A pandas Series of timestamps for the future periods you want to predict.

```python
import pandas as pd

# Load your data
df = pd.read_csv("./data/XSHG_5min_600977.csv")
df['timestamps'] = pd.to_datetime(df['timestamps'])

# Define context window and prediction length
lookback = 400
pred_len = 120

# Prepare inputs for the predictor
x_df = df.loc[:lookback-1, ['open', 'high', 'low', 'close', 'volume', 'amount']]
x_timestamp = df.loc[:lookback-1, 'timestamps']
y_timestamp = df.loc[lookback:lookback+pred_len-1, 'timestamps']
```

#### 4. Generate Forecasts 

Call the `predict` method to generate forecasts. You can control the sampling process with parameters like `T`, `top_p`, and `sample_count` for probabilistic forecasting.

```python
# Generate predictions
pred_df = predictor.predict(
    df=x_df,
    x_timestamp=x_timestamp,
    y_timestamp=y_timestamp,
    pred_len=pred_len,
    T=1.0,          # Temperature for sampling
    top_p=0.9,      # Nucleus sampling probability
    sample_count=1  # Number of forecast paths to generate and average
)

print("Forecasted Data Head:")
print(pred_df.head())
```

The `predict` method returns a pandas DataFrame containing the forecasted values for `open`, `high`, `low`, `close`, `volume`, and `amount`, indexed by the `y_timestamp` you provided.

For efficient processing of multiple time series, Kronos provides a `predict_batch` method that enables parallel prediction on multiple datasets simultaneously. This is particularly useful when you need to forecast multiple assets or time periods at once.

```python
# Prepare multiple datasets for batch prediction
df_list = [df1, df2, df3]  # List of DataFrames
x_timestamp_list = [x_ts1, x_ts2, x_ts3]  # List of historical timestamps
y_timestamp_list = [y_ts1, y_ts2, y_ts3]  # List of future timestamps

# Generate batch predictions
pred_df_list = predictor.predict_batch(
    df_list=df_list,
    x_timestamp_list=x_timestamp_list,
    y_timestamp_list=y_timestamp_list,
    pred_len=pred_len,
    T=1.0,
    top_p=0.9,
    sample_count=1,
    verbose=True
)

# pred_df_list contains prediction results in the same order as input
for i, pred_df in enumerate(pred_df_list):
    print(f"Predictions for series {i}:")
    print(pred_df.head())
```

**Important Requirements for Batch Prediction:**
- All series must have the same historical length (lookback window)
- All series must have the same prediction length (`pred_len`)
- Each DataFrame must contain the required columns: `['open', 'high', 'low', 'close']`
- `volume` and `amount` columns are optional and will be filled with zeros if missing

The `predict_batch` method leverages GPU parallelism for efficient processing and automatically handles normalization and denormalization for each series independently.

#### 5. Example and Visualization

For a complete, runnable script that includes data loading, prediction, and plotting, please see [`examples/prediction_example.py`](examples/prediction_example.py).

Running this script will generate a plot comparing the ground truth data against the model's forecast, similar to the one shown below:

<p align="center">
    <img src="figures/prediction_example.png" alt="Forecast Example" align="center" width="600px" />
</p>

Additionally, we provide a script that makes predictions without Volume and Amount data, which can be found in [`examples/prediction_wo_vol_example.py`](examples/prediction_wo_vol_example.py).


## 🔧 Finetuning on Your Own Data (A-Share Market Example)

We provide a complete pipeline for finetuning Kronos on your own datasets. As an example, we demonstrate how to use [Qlib](https://github.com/microsoft/qlib) to prepare data from the Chinese A-share market and conduct a simple backtest.

> **Disclaimer:** This pipeline is intended as a demonstration to illustrate the finetuning process. It is a simplified example and not a production-ready quantitative trading system. A robust quantitative strategy requires more sophisticated techniques, such as portfolio optimization and risk factor neutralization, to achieve stable alpha.

The finetuning process is divided into four main steps:

1.  **Configuration**: Set up paths and hyperparameters.
2.  **Data Preparation**: Process and split your data using Qlib.
3.  **Model Finetuning**: Finetune the Tokenizer and the Predictor models.
4.  **Backtesting**: Evaluate the finetuned model's performance.

### Prerequisites

1.  First, ensure you have all dependencies from `requirements.txt` installed.
2.  This pipeline relies on `qlib`. Please install it:
    ```shell
      pip install pyqlib
    ```
3.  You will need to prepare your Qlib data. Follow the [official Qlib guide](https://github.com/microsoft/qlib) to download and set up your data locally. The example scripts assume you are using daily frequency data.

### Step 1: Configure Your Experiment

All settings for data, training, and model paths are centralized in `finetune/config.py`. Before running any scripts, please **modify the following paths** according to your environment:

*   `qlib_data_path`: Path to your local Qlib data directory.
*   `dataset_path`: Directory where the processed train/validation/test pickle files will be saved.
*   `save_path`: Base directory for saving model checkpoints.
*   `backtest_result_path`: Directory for saving backtesting results.
*   `pretrained_tokenizer_path` and `pretrained_predictor_path`: Paths to the pre-trained models you want to start from (can be local paths or Hugging Face model names).

You can also adjust other parameters like `instrument`, `train_time_range`, `epochs`, and `batch_size` to fit your specific task. If you don't use [Comet.ml](https://www.comet.com/), set `use_comet = False`.

### Step 2: Prepare the Dataset

Run the data preprocessing script. This script will load raw market data from your Qlib directory, process it, split it into training, validation, and test sets, and save them as pickle files.

```shell
python finetune/qlib_data_preprocess.py
```

After running, you will find `train_data.pkl`, `val_data.pkl`, and `test_data.pkl` in the directory specified by `dataset_path` in your config.

### Step 3: Run the Finetuning

The finetuning process consists of two stages: finetuning the tokenizer and then the predictor. Both training scripts are designed for multi-GPU training using `torchrun`.

#### 3.1 Finetune the Tokenizer

This step adjusts the tokenizer to the data distribution of your specific domain.

```shell
# Replace NUM_GPUS with the number of GPUs you want to use (e.g., 2)
torchrun --standalone --nproc_per_node=NUM_GPUS finetune/train_tokenizer.py
```

The best tokenizer checkpoint will be saved to the path configured in `config.py` (derived from `save_path` and `tokenizer_save_folder_name`).

#### 3.2 Finetune the Predictor

This step finetunes the main Kronos model for the forecasting task.

```shell
# Replace NUM_GPUS with the number of GPUs you want to use (e.g., 2)
torchrun --standalone --nproc_per_node=NUM_GPUS finetune/train_predictor.py
```

The best predictor checkpoint will be saved to the path configured in `config.py`.

### Step 4: Evaluate with Backtesting

Finally, run the backtesting script to evaluate your finetuned model. This script loads the models, performs inference on the test set, generates prediction signals (e.g., forecasted price change), and runs a simple top-K strategy backtest.

```shell
# Specify the GPU for inference
python finetune/qlib_test.py --device cuda:0
```

The script will output a detailed performance analysis in your console and generate a plot showing the cumulative return curves of your strategy against the benchmark, similar to the one below:

<p align="center">
    <img src="figures/backtest_result_example.png" alt="Backtest Example" align="center" width="700px" />
</p>

### 💡 From Demo to Production: Important Considerations

*   **Raw Signals vs. Pure Alpha**: The signals generated by the model in this demo are raw predictions. In a real-world quantitative workflow, these signals would typically be fed into a portfolio optimization model. This model would apply constraints to neutralize exposure to common risk factors (e.g., market beta, style factors like size and value), thereby isolating the **"pure alpha"** and improving the strategy's robustness.
*   **Data Handling**: The provided `QlibDataset` is an example. For different data sources or formats, you will need to adapt the data loading and preprocessing logic.
*   **Strategy and Backtesting Complexity**: The simple top-K strategy used here is a basic starting point. Production-level strategies often incorporate more complex logic for portfolio construction, dynamic position sizing, and risk management (e.g., stop-loss/take-profit rules). Furthermore, a high-fidelity backtest should meticulously model transaction costs, slippage, and market impact to provide a more accurate estimate of real-world performance.

> **📝 AI-Generated Comments**: Please note that many of the code comments within the `finetune/` directory were generated by an AI assistant (Gemini 2.5 Pro) for explanatory purposes. While they aim to be helpful, they may contain inaccuracies. We recommend treating the code itself as the definitive source of logic.

## 📖 Citation

If you use Kronos in your research, we would appreciate a citation to our [paper](https://arxiv.org/abs/2508.02739):

```
@misc{shi2025kronos,
      title={Kronos: A Foundation Model for the Language of Financial Markets}, 
      author={Yu Shi and Zongliang Fu and Shuo Chen and Bohan Zhao and Wei Xu and Changshui Zhang and Jian Li},
      year={2025},
      eprint={2508.02739},
      archivePrefix={arXiv},
      primaryClass={q-fin.ST},
      url={https://arxiv.org/abs/2508.02739}, 
}
```

## 📜 License 
This project is licensed under the [MIT License](./LICENSE).










