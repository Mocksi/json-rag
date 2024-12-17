import hashlib
import os
from datetime import datetime
from dateutil.parser import parse

def compute_file_hash(filepath):
    hasher = hashlib.sha256()
    with open(filepath, 'rb') as f:
        buf = f.read()
        hasher.update(buf)
    return hasher.hexdigest()

def parse_timestamp(timestamp_str):
    if not timestamp_str:
        return None
    # Try common formats, then fallback to dateutil
    formats = [
        '%Y-%m-%dT%H:%M:%S',
        '%Y-%m-%d %H:%M:%S',
        '%Y/%m/%d %H:%M:%S',
        '%Y-%m-%d'
    ]
    for fmt in formats:
        try:
            return datetime.strptime(timestamp_str, fmt)
        except ValueError:
            continue
    try:
        return parse(timestamp_str)
    except:
        return None

def classify_path(path):
    if path == 'root':
        return 'root'
    if '[' in path:
        return 'array_access'
    if '.' in path:
        return 'nested_object'
    return 'direct_access'

def get_json_files():
    from app.config import DATA_DIR
    import glob
    return glob.glob(os.path.join(DATA_DIR, "*.json"))
