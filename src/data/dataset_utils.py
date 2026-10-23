from pathlib import Path
import hashlib

RAW_DIR = Path('data/raw')
PROCESSED_DIR = Path('data/processed')
REPORT_DIR = Path('reports')
CLASS_NAMES = ['Glioma', 'Meningioma', 'No Tumor', 'Pituitary']
IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}


def class_from_path(path):
    aliases = {
        'glioma': 0, 'meningioma': 1,
        'notumor': 2, 'notumour': 2, 'normal': 2,
        'pituitary': 3,
    }
    for part in path.parts:
        key = part.lower().replace(' ', '').replace('_', '').replace('-', '')
        if key in aliases:
            return aliases[key]
    return None


def find_images():
    return sorted(p for p in RAW_DIR.rglob('*')
                  if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS)


def label_for_image(image):
    # Supports image.jpg + image.txt, or images/... + labels/...
    candidates = [image.with_suffix('.txt')]
    parts = list(image.parts)
    for index, part in enumerate(parts):
        if part.lower() == 'images':
            replaced = parts.copy()
            replaced[index] = 'labels'
            candidates.append(Path(*replaced).with_suffix('.txt'))
    candidates.append(image.parent / 'labels' / (image.stem + '.txt'))
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return None


def read_yolo_labels(path):
    if path is None:
        return [], ['missing_annotation']
    boxes, errors = [], []
    for line_number, line in enumerate(path.read_text(encoding='utf-8-sig').splitlines(), 1):
        if not line.strip():
            continue
        parts = line.split()
        if len(parts) != 5:
            errors.append(f'line_{line_number}:expected_5_columns')
            continue
        try:
            category = int(parts[0])
            x, y, w, h = [float(value) for value in parts[1:]]
        except ValueError:
            errors.append(f'line_{line_number}:non_numeric_value')
            continue
        if category not in range(len(CLASS_NAMES)):
            errors.append(f'line_{line_number}:invalid_class_id')
        elif not (0 <= x <= 1 and 0 <= y <= 1 and 0 < w <= 1 and 0 < h <= 1):
            errors.append(f'line_{line_number}:invalid_coordinates')
        elif x - w / 2 < -1e-6 or x + w / 2 > 1 + 1e-6 or y - h / 2 < -1e-6 or y + h / 2 > 1 + 1e-6:
            errors.append(f'line_{line_number}:box_outside_image')
        else:
            boxes.append((category, x, y, w, h))
    return boxes, errors


def file_hash(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()
