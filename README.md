# Market Microstructure Toolkit

Lightweight research toolkit to **record crypto order book snapshots** and compute core **microstructure features** such as:

- Spread & midprice  
- Depth imbalance (L1, depth-K)  
- OFI (order flow imbalance, WIP)  
- Realized variance (WIP)  
- Microprice (WIP)  

Built for hands-on learning, reproducible experiments, and quick analysis.

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)  
![Python](https://img.shields.io/badge/python-3.10%2B-blue)  
![Project](https://img.shields.io/badge/status-WIP-informational)

---

## Features

- **Exchange bootstrap** (via [ccxt](https://github.com/ccxt/ccxt))  
- **Symbol validation** (spot / swap / future)  
- **Normalized snapshots** (L1 / L2) → flat schema  
- **Recorder**: save to CSV or Parquet at fixed Hz  
- **Metrics pipeline**: enrich recordings with spread, mid, imbalance, …  
- **Logging**: every run logs to `logs/` for reproducibility  
- **Tests**: Pytest-based unit tests for core modules  

---

## Install

```bash
# clone & enter repo
git clone https://github.com/<yourname>/market-microstructure-toolkit.git
cd market-microstructure-toolkit

# create env (example with conda)
conda create -n mmtoolkit python=3.9 -y
conda activate mmtoolkit

# install package (editable mode)
pip install -e .
# optional: Parquet support
pip install pyarrow  # or fastparquet
```

# Usage(CLI)
1) Record order books
```bash
# Bybit ETH perpetual, depth 100, 30s @ 2Hz → Parquet
python -m market_microstructure_toolkit.record \
  --exchange bybit \
  --market-type swap \
  --symbol ETH/USDT:USDT \
  --seconds 30 \
  --hz 2 \
  --depth 100 \
  --book-level L2 \
  --format parquet \
  --out data/ETHUSDT_swap_30s.parquet
  ```

2) Compute metrics
```bash
python -m market_microstructure_toolkit.metrics_cli data/ETHUSDT_swap_30s.parquet 100
# → writes: data/ETHUSDT_swap_30s_metrics.csv
```

3) One-liner pipeline
```bash
python -m market_microstructure_toolkit.record \
  --exchange bybit --market-type swap \
  --symbol ETH/USDT:USDT --seconds 30 --hz 2 --depth 100 \
  --book-level L2 --format parquet --out data/ETHUSDT_swap_30s.parquet \
&& python -m market_microstructure_toolkit.metrics_cli data/ETHUSDT_swap_30s.parquet 100
```
---

4) Plotting Metrics

The toolkit now includes a **plotting CLI** that visualizes spreads, microprice, OFI, and realized variance directly from recorded CSV/Parquet files.

### Example

```bash
# Record 60s of ETH/USDT order book at depth 100
python -m market_microstructure_toolkit.record \
  --exchange bybit --market-type swap \
  --symbol ETH/USDT:USDT --seconds 60 --hz 1 --depth 100 \
  --book-level L2 --format parquet --out data/ETH_bybit_L2_60s.parquet

# Plot metrics (saves PNGs in the same folder)
python -m market_microstructure_toolkit.plot_cli data/ETH_bybit_L2_60s.parquet --depth 100 --save
# → saves PNGs into: plots/ETH_bybit_L2_60s/
```

### Mid vs Microprice
Captures the mid-quote vs microprice (liquidity-weighted mid).  
![Mid vs Microprice](https://raw.githubusercontent.com/Gruntrexpewrus/market-microstructure-toolkit/main/plots/ETH_bybit_L2_60s/mid_micro.png)

### Relative Spread (bps)
Spread relative to midprice (basis points).  
![Spread (bps)](https://raw.githubusercontent.com/Gruntrexpewrus/market-microstructure-toolkit/main/plots/ETH_bybit_L2_60s/spread_bps.png)

### Order Flow Imbalance (L1)
Flow imbalance from the best bid/ask sizes (instantaneous and cumulative).  
![OFI (L1)](https://raw.githubusercontent.com/Gruntrexpewrus/market-microstructure-toolkit/main/plots/ETH_bybit_L2_60s/ofi.png)  
![OFI (L1) Cumulative](https://raw.githubusercontent.com/Gruntrexpewrus/market-microstructure-toolkit/main/plots/ETH_bybit_L2_60s/ofi_cum.png)

### Depth-K OFI (K=10)
Order flow imbalance measured across top-10 levels (size-based and notional).  
![OFI K=10 (Size)](https://raw.githubusercontent.com/Gruntrexpewrus/market-microstructure-toolkit/main/plots/ETH_bybit_L2_60s/ofi_k10_size.png)  
![OFI K=10 (Size, Cumulative)](https://raw.githubusercontent.com/Gruntrexpewrus/market-microstructure-toolkit/main/plots/ETH_bybit_L2_60s/ofi_k10_size_cum.png)  
![OFI K=10 (Notional)](https://raw.githubusercontent.com/Gruntrexpewrus/market-microstructure-toolkit/main/plots/ETH_bybit_L2_60s/ofi_k10_notional.png)  
![OFI K=10 (Notional, Cumulative)](https://raw.githubusercontent.com/Gruntrexpewrus/market-microstructure-toolkit/main/plots/ETH_bybit_L2_60s/ofi_k10_notional_cum.png)

### Realized Variance
Rolling realized variance of midprice (RV-20).  
![Realized Variance](https://raw.githubusercontent.com/Gruntrexpewrus/market-microstructure-toolkit/main/plots/ETH_bybit_L2_60s/rv.png)

## Roadmap

**Phase 1 (done):**  
✅ Exchange bootstrap & symbol checks  
✅ Normalized snapshots (L1/L2)  
✅ Recorder (CSV/Parquet)  
✅ Metrics: spread, mid, imbalance (L1, depth-K)  

**Phase 2 (next):**  
✅ Relative spread (bps)  
✅ Microprice & microprice imbalance  
✅ Rolling realized variance & volatility proxies  
✅ Notional depth, book slope/convexity  
✅ Order flow imbalance (OFI)  

**Phase 3 (later):**  
⬜ Websocket streaming collectors  
⬜ Event-time metrics (per book update)  
⬜ Advanced visualization & analytics  
⬜ Market impact helpers (VWAP/TWAP)  
⬜ `console_scripts` entry points (`mmt-record`, `mmt-metrics`)  

---

## License

MIT License – see [LICENSE](LICENSE) for details.

---

## Contact

- 📧 Email: leonardoplacidi@gmail.com  
- 💼 LinkedIn: [Leonardo Placidi](https://www.linkedin.com/in/leonardo-p-570616198/)  
