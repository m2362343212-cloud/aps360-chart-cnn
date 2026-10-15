# Learning Relative Stock Performance from Price-Chart Images

APS360 (Applied Fundamentals of Deep Learning, University of Toronto, Fall 2026) — individual course project by Anran Miao.

> **Status:** Proposal stage (October 2026). No code or results yet; this repository will be filled in as the project progresses.

## Goal

For each S&P 500 stock and week, a convolutional neural network (CNN) looks at an image of the past 20 trading days of prices and volume, drawn next to the market index, and predicts whether the stock will **beat the median S&P 500 stock over the next 5 trading days**.

The question is whether a CNN can learn a useful short-term price signal by itself, compared with hand-made signals such as short-term reversal.

## Planned approach

| Step | Plan |
|---|---|
| Data | Daily OHLCV prices (Yahoo Finance via `yfinance`) for S&P 500 stocks, January 2000 – December 2024, plus SPY; historical index membership rebuilt from Wikipedia's list of index changes |
| Cleaning | Split/dividend-adjusted prices, NYSE trading calendar, drop windows with missing days or bad bars, point-in-time index membership |
| Input | 2 × 64 × 60 image per sample: channel 1 = stock OHLC bars, moving average and volume; channel 2 = market (SPY) path on the same scale |
| Label | 1 if the stock's next-5-day return is above that week's cross-sectional median, else 0 |
| Split | By time, no shuffling: train 2001–2014, validation 2015–2017, test 2018–2024 |
| Model | CNN with 3 conv blocks (Conv 5×3 → BatchNorm → LeakyReLU → MaxPool 2×1) and a fully connected output, trained in PyTorch |
| Baselines | Short-term reversal rule; logistic regression on 6 price/volume features; MLP on the same 20-day numbers |
| Evaluation | Test accuracy and ROC-AUC; weekly long–short portfolio (top vs. bottom 10%) Sharpe ratio before and after trading costs |

## Planned repository structure

```
data/          # download and cleaning scripts (raw data is not committed)
src/           # image rendering, dataset, models, training
notebooks/     # Colab notebooks for experiments
reports/       # proposal, progress report, final report
```

## Data note

Price data comes from Yahoo Finance and is not redistributed in this repository. Scripts will download it directly.

## Disclaimer

This is a course research project, not investment advice.

## Key reference

Jiang, J., Kelly, B., & Xiu, D. (2023). (Re-)Imag(in)ing Price Trends. *The Journal of Finance*, 78(6), 3193–3249.
