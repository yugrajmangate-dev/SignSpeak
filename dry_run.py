"""Validate recorded sessions without training or generating synthetic data."""
from check_data import check_data
from common import ROOT

if __name__ == '__main__':
    raise SystemExit(0 if check_data(ROOT / 'data') else 1)
