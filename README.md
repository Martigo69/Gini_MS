# Gini_MS

Privacy-preserving feature selection using the MS-GINI protocol implemented in MP-SPDZ.

This repository contains an end-to-end workflow to:
- Prepare dataset inputs for MP-SPDZ from CSV files.
- Run secure MS-GINI feature scoring in MPC.
- Reveal and inspect feature scores, thresholds, and selected feature indices.
- Validate outputs with a local Python simulation.

## Project Overview

The core MPC program is implemented in `gini_fs.mpc` and follows:
- Protocol 2 (MS_GINI): Secure GINI score computation per feature.
- Protocol 1 (FILTER_FS): Secure top-k feature selection.
- Protocol 3 (GINI-FS): End-to-end flow over all features.

The program prints:
- Per-feature MS-GINI score details.
- Theta (feature mean threshold).
- Partition counts (`a`, `b`) and partition impurity terms.
- Top-k selected feature indices.

## Repository Structure

- `gini_fs.mpc`: Main MP-SPDZ implementation of MS-GINI and secure filtering.
- `client.py`: Converts CSV datasets into MP-SPDZ player input format (`Player-Data/Input-P0-0-<dataset>`).
- `simulate.py`: Plain Python reference implementation for sanity checks.
- `Player-Data/`: Example input files generated for MP-SPDZ runs.

## Dataset Configuration

Dataset-specific metadata is configured in `gini_fs.mpc`.

| Dataset | CSV File | Target Column | Dropped Columns | Rows (m) | Features (p) | Classes (n) | Selected Features (k) |
|---|---|---|---|---:|---:|---:|---:|
| beans | `beans_kmeans.csv` | `Class` | none | 13611 | 16 | 7 | 10 |
| diabetes | `diabetes_kmeans.csv` | `Outcome` | none | 768 | 8 | 2 | 7 |
| divorce | `divorce.csv` | `Class` | none | 170 | 54 | 2 | 45 |
| parkinsons | `parkinsons_kmeans.csv` | `status` | `name` | 195 | 22 | 2 | 18 |
| rice | `rice_binned_kmeans.csv` | `Class` | none | 3810 | 7 | 2 | 5 |
| wdbc | `wdbc_binned_kmeans.csv` | `Diagnosis` | `ID` | 569 | 30 | 2 | 22 |

## Prerequisites

- MP-SPDZ installed and built.
- Python 3.9+.
- Python packages:
  - numpy
  - pandas
  - scikit-learn

Install Python dependencies:

```bash
pip install numpy pandas scikit-learn
```

## Quick Start

### 1. Choose a dataset

Set the `dataset` variable near the top of `gini_fs.mpc`.

### 2. Generate MPC input file

In `client.py`, set `dataset_name`, then run:

```bash
python client.py
```

This writes `Player-Data/Input-P0-0-<dataset>`.

### 3. Copy files into MP-SPDZ and rename player input

After generating `Player-Data/Input-P0-0-<dataset>` in this repository, copy files to your MP-SPDZ repository:

```bash
# from Gini_MS repo root
cp gini_fs.mpc /path/to/MP-SPDZ/Programs/Source/gini_fs.mpc
cp Player-Data/Input-P0-0-<dataset> /path/to/MP-SPDZ/Player-Data/Input-P0-0
```

`gini_fs.mpc` must be placed in `Programs/Source/`, and the input file must be renamed to `Input-P0-0` in MP-SPDZ.

### 4. Compile and run with active 3PC

From your MP-SPDZ root:

```bash
./compile.py -R 64 gini_fs
Scripts/ring.sh gini_fs -N 3 -v
```

This uses ring with 3 parties (active 3PC setting in your workflow).

### 5. Validate with local simulation (optional)

Set `dataset_name` in `simulate.py`, then run:

```bash
python simulate.py
```

## Output Summary

A successful run prints:
- Data loading status.
- MS-GINI score table for all features.
- Top-k selected feature indices.
- MP-SPDZ detailed cost metrics (online/offline rounds, multiplications, openings, communication volume).

## Reproducibility Notes

- Keep dataset preprocessing consistent between `client.py`, `simulate.py`, and `gini_fs.mpc`.
- Ensure feature ordering in CSV files is unchanged across runs.
- For fair cross-dataset comparisons, keep protocol, precision, and MP-SPDZ runtime flags fixed.

## License

No license file is currently included in this repository. Add one before public distribution if required.
