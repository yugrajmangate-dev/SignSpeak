"""Train on earlier sessions; select by last-session macro F1 (tie: accuracy)."""
import argparse
import json
import time
from pathlib import Path
import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, ConfusionMatrixDisplay, f1_score
from sklearn.neural_network import MLPClassifier
from common import ROOT, FEATURES
from check_data import check_data


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, default=ROOT / 'data')
    parser.add_argument('--output-dir', type=Path, default=ROOT)
    parser.add_argument('--signers', type=int, default=1)
    args = parser.parse_args()
    if args.signers < 1:
        parser.error('--signers must be a positive integer.')
    if not check_data(args.data_dir):
        parser.exit(1, 'Training refused: recorded data must pass check_data.py.\n')
    files = sorted(args.data_dir.glob('session_*.csv'), key=lambda p: int(p.stem.split('_')[-1]))
    if len(files) < 2:
        parser.error('Need at least two sessions; run collect.py twice. No random frame split.')
    frames = [pd.read_csv(p, dtype={'label': str}) for p in files]
    for path, frame in zip(files, frames):
        if frame.empty or list(frame.columns) != FEATURES + ['label'] or frame.label.isna().any():
            parser.error(f'{path}: expected nonempty CSV with 42 ordered features and label.')
        if not np.isfinite(frame[FEATURES].to_numpy(dtype=float)).all():
            parser.error(f'{path}: features must be finite numbers.')
    train, test = pd.concat(frames[:-1]), frames[-1]
    x, xt = train[FEATURES].to_numpy(), test[FEATURES].to_numpy()
    if train.label.nunique() < 2:
        parser.error('Training sessions need at least two classes.')
    labels = sorted(set(train.label) | set(test.label))
    models = {'RandomForest': RandomForestClassifier(n_estimators=100, random_state=42),
              'MLP': MLPClassifier(hidden_layer_sizes=(32,), max_iter=500, random_state=42)}
    scores, predictions = {}, {}
    for name, model in models.items():
        model.fit(x, train.label)
        pred = model.predict(xt)
        predictions[name] = pred
        model.predict(xt[:1])  # warm-up; measure model-only single-sample latency
        start = time.perf_counter()
        for _ in range(50):
            model.predict(xt[:1])
        latency = (time.perf_counter() - start) * 1000 / 50
        scores[name] = {'accuracy': float(accuracy_score(test.label, pred)),
                        'macro_f1': float(f1_score(test.label, pred, labels=labels, average='macro', zero_division=0)),
                        'latency_ms': latency}
    winner = max(scores, key=lambda name: (scores[name]['macro_f1'], scores[name]['accuracy']))
    metrics = {'data_source': 'recorded', 'n_sessions': len(files),
               'n_samples': len(train) + len(test), 'labels': labels,
               'test_session': files[-1].name, 'train_size': len(train), 'test_size': len(test),
               'signers': args.signers, 'model_chosen': winner, **scores[winner],
               'train_sessions': [p.name for p in files[:-1]], 'all_models': scores,
               'evaluation_note': 'Last session also selects winner; scores are not an independent final evaluation.',
               'per_class_report': classification_report(test.label, predictions[winner], labels=labels, output_dict=True, zero_division=0)}
    results = args.output_dir / 'results'
    results.mkdir(parents=True, exist_ok=True)
    joblib.dump({'model': models[winner], 'features': FEATURES, 'data_source': 'recorded'}, args.output_dir / 'model.joblib')
    (results / 'metrics.json').write_text(json.dumps(metrics, indent=2))
    ConfusionMatrixDisplay.from_predictions(test.label, predictions[winner], labels=labels, xticks_rotation=45, cmap='Blues')
    plt.tight_layout()
    plt.savefig(results / 'confusion_matrix.png', dpi=120)
    plt.close()
    print(json.dumps(metrics, indent=2))


if __name__ == '__main__':
    main()
