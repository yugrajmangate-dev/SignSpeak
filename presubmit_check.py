"""Report submission readiness; --deep installs cloud deps in a fresh temp venv."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from PIL import Image

ROOT = Path(__file__).resolve().parent
SECTIONS = ('Team name and members', 'Problem', 'Solution', 'Pipeline', 'Tech stack', 'Setup',
            'Run locally', 'Screenshots', 'Metrics', 'Limitations and responsible use', 'AI tools used')


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--deep', action='store_true')
    args = parser.parse_args()
    checks = []
    def check(name: str, passed: bool, detail: str = '') -> None:
        checks.append(passed)
        print(f'{"PASS" if passed else "FAIL"}: {name}' + (f' — {detail}' if detail else ''), flush=True)
    readme = (ROOT / 'README.md').read_text() if (ROOT / 'README.md').exists() else ''
    missing = [s for s in SECTIONS if f'## {s}' not in readme]
    check('Required README sections', not missing and 'FT-04' in readme, ', '.join(missing))
    ai = readme.partition('## AI tools')[2]
    check('Codex AI-tools disclosure', 'Codex' in ai.split('\n## ')[0])
    docs = [ROOT / 'README.md', *sorted((ROOT / 'docs').rglob('*.md'))]
    markers = ('TO'+'DO', 'FIX'+'ME', '<EDIT'+' ME>', '[Placeholder', '[Team name]', '[Name ')
    hits = [p.name for p in docs if p.exists() and any(t in p.read_text() for t in markers)]
    check('No documentation placeholders', not hits, ', '.join(hits))
    try:
        m = json.loads((ROOT / 'results/metrics.json').read_text())
        recorded = m.get('data_source') == 'recorded'
        if m.get('data_source') == 'pretrained':
            model = ROOT / 'models/gesture_recognizer.task'
            recorded = model.exists() and hashlib.sha256(model.read_bytes()).hexdigest() == m.get('model_sha256') and all(m.get(key) is None for key in ('accuracy', 'macro_f1', 'latency_ms'))
    except (OSError, ValueError):
        recorded = False
    check('Metrics provenance: recorded evaluation or disclosed unevaluated pretrained model', recorded)
    for name in ('live_demo.png', 'streamlit_app.png'):
        check(f'Screenshot {name}', (ROOT / 'docs/screenshots' / name).is_file())
    thumbnail = ROOT / 'docs/thumbnail.png'
    try:
        with Image.open(thumbnail) as image:
            valid = image.width * 2 == image.height * 3 and thumbnail.stat().st_size < 5_000_000
    except (OSError, ValueError):
        valid = False
    check('Thumbnail exists, is 3:2 and under 5 MB', valid)
    story = ROOT / 'docs/devpost_story.md'
    check('Devpost story exists', story.is_file())
    try:
        proc = subprocess.run(['gh', 'repo', 'view', '--json', 'visibility'], cwd=ROOT, capture_output=True, text=True, timeout=20)
        public = proc.returncode == 0 and json.loads(proc.stdout).get('visibility') == 'PUBLIC'
    except (OSError, ValueError, subprocess.TimeoutExpired):
        public = False
    check('GitHub repository is public', public)
    if args.deep:
        try:
            with tempfile.TemporaryDirectory(prefix='signspeak-deep-') as folder:
                env = {**os.environ, 'PYTHONDONTWRITEBYTECODE': '1'}
                subprocess.run([sys.executable, '-B', '-m', 'venv', folder], check=True, timeout=120, env=env)
                python = str(Path(folder) / ('Scripts/python.exe' if os.name == 'nt' else 'bin/python'))
                subprocess.run([python, '-m', 'pip', 'install', '-r', str(ROOT / 'requirements.txt')], check=True, timeout=600, env=env)
                subprocess.run([python, '-B', '-c', 'import app'], check=True, cwd=ROOT, timeout=120, env=env)
            check('Fresh environment cloud install and import app', True)
        except (OSError, subprocess.SubprocessError) as error:
            check('Fresh environment cloud install and import app', False, str(error))
    passed = sum(checks)
    print(f'Summary: {passed}/{len(checks)} PASS; {len(checks)-passed} FAIL.', flush=True)
    raise SystemExit(0 if all(checks) else 1)


if __name__ == '__main__':
    main()
