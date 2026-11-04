"""Print per-session label counts and reject insufficient or mismatched data."""
import argparse
from pathlib import Path
import pandas as pd
from common import ROOT


def check_data(folder: Path) -> bool:
    files = list(folder.glob('session_*.csv'))
    try:
        files.sort(key=lambda p: int(p.stem.split('_')[-1]))
    except ValueError:
        print('FAIL: session filenames must end in a numeric ID.')
        return False
    counts = {}
    for path in files:
        try:
            frame = pd.read_csv(path, dtype={'label': str})
            if 'label' not in frame or frame.label.isna().any():
                raise ValueError('missing or blank labels')
            counts[path.name] = frame.label.value_counts()
        except (OSError, ValueError, pd.errors.ParserError) as error:
            print(f'FAIL: {path.name}: {error}')
            return False
    if counts:
        table = pd.DataFrame(counts).fillna(0).astype(int).T.sort_index(axis=1)
        print('Frames per session per label:\n' + table.to_string())
    else:
        table = pd.DataFrame()
        print('No session CSV files found.')
    errors = []
    if len(files) < 2:
        errors.append('At least two session files are required.')
    if table.empty or (table < 60).any().any():
        errors.append('Every label needs at least 60 frames in every session.')
    if counts and any(set(c.index) != set(next(iter(counts.values())).index) for c in counts.values()):
        errors.append('Labels differ between sessions.')
    for error in errors:
        print('FAIL: ' + error)
    if not errors:
        print('PASS: data requirements satisfied.')
    return not errors


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, default=ROOT / 'data')
    raise SystemExit(0 if check_data(parser.parse_args().data_dir) else 1)
