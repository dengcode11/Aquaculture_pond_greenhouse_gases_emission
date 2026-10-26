from pathlib import Path
import json
import os


ROOT = Path(__file__).resolve().parents[2]
WORKSPACE = Path(os.environ.get('AQUACULTURE_WORKSPACE', ROOT / 'outputs/research_workspace')).resolve()


def mappings():
    config = os.environ.get('AQUACULTURE_EXTERNAL_CONFIG')
    if not config:
        return {}
    filename = Path(config).resolve()
    content = json.loads(filename.read_text(encoding='utf-8-sig'))
    return {key.strip('/'): Path(value) if Path(value).is_absolute() else filename.parent / value for key, value in content.items()}


def project_path(relative):
    relative = str(relative).replace('\\', '/').strip('/')
    for key, value in sorted(mappings().items(), key=lambda item: len(item[0]), reverse=True):
        if relative == key or relative.startswith(key + '/'):
            return (value / relative[len(key):].strip('/')).resolve()
    return WORKSPACE / relative


def external_path(relative):
    relative = str(relative).replace('\\', '/').strip('/')
    for key, value in sorted(mappings().items(), key=lambda item: len(item[0]), reverse=True):
        if relative == key or relative.startswith(key + '/'):
            return (value / relative[len(key):].strip('/')).resolve()
    raise FileNotFoundError(f'External input not configured: {relative}. Pass --external-config to Scripts/run.py.')
