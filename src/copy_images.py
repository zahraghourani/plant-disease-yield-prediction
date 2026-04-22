import os
import shutil
import random

src = r'data\Crop___DIsease'
dst = r'detection_data\images'

os.makedirs(dst, exist_ok=True)

for class_name in os.listdir(src):
    class_path = os.path.join(src, class_name)
    if not os.path.isdir(class_path):
        continue
    images = [f for f in os.listdir(class_path) if f.lower().endswith(('.jpg','.jpeg','.png'))]
    random.seed(42)
    selected = random.sample(images, min(20, len(images)))
    for img in selected:
        shutil.copy(os.path.join(class_path, img), os.path.join(dst, f'{class_name}_{img}'))

print('Done. Copied', sum(1 for _ in os.listdir(dst)), 'images.')