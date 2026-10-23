from collections import Counter
import csv
import random

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw

from src.data.dataset_utils import (
    RAW_DIR, REPORT_DIR, CLASS_NAMES, find_images,
    class_from_path, label_for_image, read_yolo_labels,
)


def main():
    images = find_images()
    if not images:
        raise SystemExit(f'No images found in {RAW_DIR}. Extract your dataset first.')

    figure_dir = REPORT_DIR / 'figures'
    metric_dir = REPORT_DIR / 'metrics'
    figure_dir.mkdir(parents=True, exist_ok=True)
    metric_dir.mkdir(parents=True, exist_ok=True)

    counts = Counter()
    rows = []
    usable = []

    for image_path in images:
        category = class_from_path(image_path.relative_to(RAW_DIR))
        counts[CLASS_NAMES[category] if category is not None else 'Unknown'] += 1
        annotation_path = label_for_image(image_path)
        boxes, errors = read_yolo_labels(annotation_path)
        try:
            with Image.open(image_path) as img:
                img.verify()
            with Image.open(image_path) as img:
                width, height = img.size
            usable.append((image_path, boxes))
            image_error = ''
        except (OSError, ValueError) as exc:
            width = height = 0
            image_error = str(exc)
        rows.append({
            'image': str(image_path),
            'class': CLASS_NAMES[category] if category is not None else 'Unknown',
            'width': width, 'height': height,
            'label': str(annotation_path) if annotation_path else '',
            'box_count': len(boxes), 'annotation_errors': ';'.join(errors),
            'image_error': image_error,
        })

    with (metric_dir / 'dataset_inventory.csv').open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    print(f'Images found: {len(images)}')
    for name, count in counts.items():
        print(f'{name}: {count}')
    print(f'Unreadable images: {sum(bool(r["image_error"]) for r in rows)}')

    plt.figure(figsize=(9, 5))
    names = CLASS_NAMES + (['Unknown'] if counts['Unknown'] else [])
    plt.bar(names, [counts[name] for name in names])
    plt.title('MRI image distribution by class')
    plt.ylabel('Number of images')
    plt.xticks(rotation=15)
    plt.tight_layout()
    plt.savefig(figure_dir / 'class_distribution.png', dpi=150)
    plt.close()

    random.Random(42).shuffle(usable)
    samples = usable[:min(12, len(usable))]
    if samples:
        fig, axes = plt.subplots(3, 4, figsize=(14, 10))
        for axis in axes.flat:
            axis.axis('off')
        for axis, (path, boxes) in zip(axes.flat, samples):
            with Image.open(path) as source:
                image = source.convert('RGB')
            drawer = ImageDraw.Draw(image)
            width, height = image.size
            for category, x, y, w, h in boxes:
                if category == 2:  # No Tumor is not a lesion.
                    continue
                box = ((x-w/2)*width, (y-h/2)*height,
                       (x+w/2)*width, (y+h/2)*height)
                drawer.rectangle(box, outline='red', width=max(2, width // 120))
            axis.imshow(image)
            axis.set_title(CLASS_NAMES[class_from_path(path.relative_to(RAW_DIR))]
                           if class_from_path(path.relative_to(RAW_DIR)) is not None
                           else 'Unknown', fontsize=10)
            axis.axis('off')
        plt.tight_layout()
        plt.savefig(figure_dir / 'sample_images.png', dpi=150)
        plt.close()

    print('Saved inventory and plots under reports/.')


if __name__ == '__main__':
    main()
