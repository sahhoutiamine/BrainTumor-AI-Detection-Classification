import csv
from collections import Counter
from PIL import Image

from src.data.dataset_utils import (
    RAW_DIR, REPORT_DIR, CLASS_NAMES, find_images,
    class_from_path, label_for_image, read_yolo_labels,
)


def audit_image(path):
    problems = []
    try:
        with Image.open(path) as img:
            img.verify()
    except (OSError, ValueError):
        problems.append('corrupt_image')

    folder_class = class_from_path(path.relative_to(RAW_DIR))
    if folder_class is None:
        problems.append('unknown_folder_class')

    label_path = label_for_image(path)
    boxes, label_errors = read_yolo_labels(label_path)
    problems.extend(label_errors)

    if label_path is not None and not boxes and not label_errors:
        if folder_class != 2:
            problems.append('empty_tumor_annotation')

    box_classes = {box[0] for box in boxes}
    if len(box_classes) > 1:
        problems.append('mixed_annotation_classes')
    if folder_class is not None and box_classes and box_classes != {folder_class}:
        problems.append('folder_label_disagreement')
    if folder_class == 2 and boxes:
        problems.append('no_tumor_has_artificial_boxes')
    if folder_class in (0, 1, 3) and not boxes:
        problems.append('tumor_without_valid_box')

    return {
        'image': str(path),
        'folder_class': CLASS_NAMES[folder_class] if folder_class is not None else 'Unknown',
        'annotation': str(label_path) if label_path else '',
        'box_count': len(boxes),
        'issues': ';'.join(dict.fromkeys(problems)),
    }


def main():
    images = find_images()
    if not images:
        raise SystemExit(f'No images found in {RAW_DIR}.')
    rows = [audit_image(path) for path in images]
    output = REPORT_DIR / 'metrics' / 'annotation_audit.csv'
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    frequencies = Counter(issue for row in rows for issue in row['issues'].split(';') if issue)
    print(f'Images audited: {len(rows)}')
    for issue, count in frequencies.most_common():
        print(f'{issue}: {count}')
    print(f'Full audit: {output}')


if __name__ == '__main__':
    main()
