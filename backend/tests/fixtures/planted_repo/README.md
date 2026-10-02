# Small MLP baselines on digits

Code for "Small Networks, Solid Baselines: An MLP Study on Handwritten Digits".

## Setup

```bash
pip install -r requirements.txt
```

## Reproduce Table 1

MLP (64 hidden units):

```bash
python train.py --out results.json
```

Logistic-regression baseline:

```bash
python baseline.py
```

The CNN in the appendix needs a GPU: `python train_gpu.py`

Tested with Python 3.10.
