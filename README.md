# Market Microstructure Toolkit

Lightweight toolkit to collect crypto order book snapshots and compute core microstructure features
(spread, depth imbalance, OFI, realized variance, microprice). Built for hands-on learning and quick
analysis.

> Status: Work in progress. **Day 1:** exchange bootstrap + symbol checks.

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![Project](https://img.shields.io/badge/status-WIP-informational)

## Tech stack
- Python • pandas • matplotlib • ccxt

## Quickstart
```bash
# create env (example with conda)
conda create -n mmtoolkit python=3.10 -y
conda activate mmtoolkit
pip install -r requirements.txt
```
Usage / Milestones
	•	✅ T2: Exchange bootstrap & symbol checks
	•	⬜ T3: Normalized top-N order book snapshot (in progress)
	•	⬜ T4: 60s recorder (1 Hz) → CSV
	•	⬜ T5–T9: Metrics (spread, DI, OFI, RV, microprice) + plots
	•	⬜ T10: One-command pipeline (record → metrics → plots)

Results (preview)

Screenshots of plots will appear here as tasks complete.

Repository hygiene
	•	MIT license • Contributor Covenant • Conventional commits • Keep a Changelog

Contact
	•	Email: leonardoplacidi@gmail.com
	•	LinkedIn: https://www.linkedin.com/in/leonardo-p-570616198/

