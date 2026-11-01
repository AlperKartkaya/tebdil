"""Reproduce the frozen unexecuted reasoning pool; never overwrite evidence."""
import argparse
import json
from pathlib import Path

from tebdil.tasks.reasoning import build_manifest, development_prompts, validate_manifest

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args()
    manifest = build_manifest()
    validate_manifest(manifest)
    public = development_prompts(manifest)
    outputs = {'configs/reasoning_v1_tasks.json':manifest,
               'site/static/evidence/reasoning_v1_development.json':public}
    serialized = {p:json.dumps(data,indent=2,allow_nan=False)+'\n' for p,data in outputs.items()}
    if args.check:
        for path,text in serialized.items():
            if (ROOT/path).read_text() != text:
                raise ValueError(f'Frozen task bytes differ: {path}')
    else:
        if any((ROOT/path).exists() for path in serialized):
            parser.error('Frozen output exists; use --check, never overwrite it')
        for path,text in serialized.items():
            with (ROOT/path).open('x') as stream:
                stream.write(text)
    print('PASS: 6 development / 12 reserved instances, deterministic prompts; no model execution')


if __name__ == '__main__':
    main()
