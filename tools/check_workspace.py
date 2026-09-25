"""Validate notebook syntax, portfolio links, and preservation of original blobs."""
import ast
import hashlib
import json
import re
from pathlib import Path
from urllib.parse import unquote, urlsplit
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]


def main():
    catalog = json.loads((ROOT/'catalog.json').read_text())
    checked = 0
    for entry in catalog:
        for key in ['notebook', 'original', 'case_study']:
            if not (ROOT/entry[key]).is_file():
                raise ValueError(f'Missing {key}: {entry[key]}')
        path = ROOT/entry['notebook']
        notebook = json.loads(path.read_text())
        assert notebook['nbformat'] == 4
        assert path.suffix == '.ipynb'
        assert notebook['cells'][0]['cell_type'] == 'markdown'
        for i, cell in enumerate(notebook['cells']):
            if cell['cell_type'] == 'code':
                ast.parse(''.join(cell['source']), filename=f'{path.name}:cell-{i}')
                assert cell['outputs'] == [], f'Stale outputs: {path}'
                assert cell['execution_count'] is None
        checked += 1
    archive = json.loads((ROOT/'archive/manifest.json').read_text())
    for entry in archive['files']:
        content = (ROOT/entry['path']).read_bytes()
        digest = hashlib.sha1(b'blob '+str(len(content)).encode()+b'\0'+content).hexdigest()
        assert digest == entry['git_blob_sha'], f'Original changed: {entry["path"]}'
    for path in ROOT.rglob('*.py'):
        if '.venv' not in path.parts:
            ast.parse(path.read_text(), filename=str(path))
    for path in ROOT.rglob('*.md'):
        if 'archive' in path.relative_to(ROOT).parts:
            continue
        for target in re.findall(r'\]\(([^\s)]+)\)', path.read_text()):
            link = urlsplit(target)
            if link.scheme or link.netloc or not link.path:
                continue
            dest = path.parent/unquote(link.path)
            assert dest.exists(), f'Broken local link in {path.name}: {target}'
    ET.parse(ROOT/'assets/cover.svg')
    print(f'PASS: {checked} notebook sources, local Markdown links, Python syntax, SVG, and {len(archive["files"])} original blobs')


if __name__ == '__main__':
    main()
