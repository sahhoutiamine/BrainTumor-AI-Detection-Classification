import csv
import shutil
from collections import Counter
from pathlib import Path

from PIL import Image
from sklearn.model_selection import train_test_split
import yaml

from src.data.dataset_utils import (
    RAW_DIR, PROCESSED_DIR, REPORT_DIR, CLASS_NAMES, find_images,
    class_from_path, label_for_image, read_yolo_labels, file_hash,
)

SEED = 42


def collect_valid_records():
    records, excluded, seen_hashes = [], [], set()
    for path in find_images():
        category = class_from_path(path.relative_to(RAW_DIR))
        reasons = []
        if category is None:
            reasons.append('unknown_folder_class')
        try:
            with Image.open(path) as image:
                image.verify()
        except (OSError, ValueError):
            reasons.append('corrupt_image')

        annotation = label_for_image(path)
        boxes, errors = read_yolo_labels(annotation)
        # Missing annotation is acceptable only for No Tumor.
        if errors and not (category == 2 and errors == ['missing_annotation']):
            reasons.extend(errors)

        classes_in_boxes = {box[0] for box in boxes}
        if category in (0, 1, 3):
            if not boxes:
                reasons.append('missing_tumor_box')
            if classes_in_boxes != {category}:
                reasons.append('class_conflict_or_mixed_boxes')
        if category == 2 and any(box[0] != 2 for box in boxes):
            reasons.append('tumor_box_in_no_tumor_image')

        if not reasons:
            fingerprint = file_hash(path)
            if fingerprint in seen_hashes:
                reasons.append('exact_duplicate_image')
            else:
                seen_hashes.add(fingerprint)

        if reasons:
            excluded.append({'image': str(path), 'reasons': ';'.join(reasons)})
        else:
            records.append({'path': path, 'class_id': category,
                            'boxes': boxes, 'hash': fingerprint})
    return records, excluded


def split_records(records):
    labels = [record['class_id'] for record in records]
    counts = Counter(labels)
    if min(counts.values()) < 10 or len(counts) != 4:
        raise ValueError('Need at least 10 valid images per class for an 80/10/10 stratified split.')

    training, temporary = train_test_split(
        records, test_size=0.20, random_state=SEED, stratify=labels
    )
    validation, testing = train_test_split(
        temporary, test_size=0.50, random_state=SEED,
        stratify=[record['class_id'] for record in temporary]
    )
    return {'train': training, 'val': validation, 'test': testing}


def export_splits(splits):
    classification_root = PROCESSED_DIR / 'classification'
    detection_root = PROCESSED_DIR / 'detection'
    if any((classification_root / part).exists() and
           any(p.name != '.gitkeep' for p in (classification_root / part).iterdir()) for part in splits):
        raise SystemExit('Classification output is not empty. Remove previous generated splits before rerunning.')
    if any((detection_root / 'images' / part).exists() and
           any(p.name != '.gitkeep' for p in (detection_root / 'images' / part).iterdir()) for part in splits):
        raise SystemExit('Detection output is not empty. Remove previous generated splits before rerunning.')

    manifest = []
    for split_name, records in splits.items():
        for record in records:
            source = record['path']
            class_id = record['class_id']
            # Hash prefix + index prevents filename collisions.
            unique_name = record['hash'][:16] + source.suffix.lower()
            class_dir = classification_root / split_name / CLASS_NAMES[class_id]
            image_dir = detection_root / 'images' / split_name
            label_dir = detection_root / 'labels' / split_name
            for directory in (class_dir, image_dir, label_dir):
                directory.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, class_dir / unique_name)
            shutil.copy2(source, image_dir / unique_name)

            # Detection uses only tumor classes: 0 Glioma, 1 Meningioma, 2 Pituitary.
            detector_ids = {0: 0, 1: 1, 3: 2}
            valid_boxes = [box for box in record['boxes'] if box[0] in detector_ids]
            content = '\n'.join(
                f'{detector_ids[box[0]]} {box[1]:.8f} {box[2]:.8f} {box[3]:.8f} {box[4]:.8f}'
                for box in valid_boxes
            )
            (label_dir / (Path(unique_name).stem + '.txt')).write_text(
                content + ('\n' if content else ''), encoding='utf-8'
            )
            manifest.append({'split': split_name, 'class': CLASS_NAMES[class_id],
                             'source': str(source), 'sha256': record['hash']})

    detection_yaml = {
        'path': str(detection_root.resolve()),
        'train': 'images/train', 'val': 'images/val', 'test': 'images/test',
        'names': {0: 'Glioma', 1: 'Meningioma', 2: 'Pituitary'},
    }
    with (detection_root / 'data.yaml').open('w', encoding='utf-8') as handle:
        yaml.safe_dump(detection_yaml, handle, sort_keys=False)

    metrics_dir = REPORT_DIR / 'metrics'
    metrics_dir.mkdir(parents=True, exist_ok=True)
    with (metrics_dir / 'split_manifest.csv').open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=manifest[0].keys())
        writer.writeheader()
        writer.writerows(manifest)
    print('Created classification and detection datasets.')
    for split_name, records in splits.items():
        print(f'{split_name}: {len(records)} images; classes={dict(Counter(r["class_id"] for r in records))}')
    print(f'Detection YAML: {detection_root / "data.yaml"}')


def main():
    records, excluded = collect_valid_records()
    if not records:
        raise SystemExit('No usable records. Review annotation_audit.csv first.')
    metrics_dir = REPORT_DIR / 'metrics'
    metrics_dir.mkdir(parents=True, exist_ok=True)
    with (metrics_dir / 'excluded_images.csv').open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=['image', 'reasons'])
        writer.writeheader()
        writer.writerows(excluded)
    print(f'Eligible unique images: {len(records)}; excluded: {len(excluded)}')
    export_splits(split_records(records))


if __name__ == '__main__':
    main()
