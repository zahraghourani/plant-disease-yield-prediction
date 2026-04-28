import os
from PIL import Image
from PIL.ExifTags import TAGS

def check_metadata(image_path):
    try:
        img = Image.open(image_path)
        exif = img._getexif()
        if not exif:
            return None
        # Map tag IDs to names
        exif_data = {}
        for tag_id, value in exif.items():
            tag = TAGS.get(tag_id, tag_id)
            exif_data[tag] = value
        return exif_data
    except Exception as e:
        return f"Error: {e}"

# Check first 5 images in detection_data/images (or any folder)
image_dir = 'detection_data/images'
sample_images = [f for f in os.listdir(image_dir) if f.lower().endswith(('.jpg','.jpeg','.png'))][:5]

for img_file in sample_images:
    path = os.path.join(image_dir, img_file)
    data = check_metadata(path)
    print(f"\n{img_file}:")
    if data is None:
        print("  No EXIF metadata found.")
    else:
        # Show only relevant tags
        relevant = ['GPSInfo', 'DateTime', 'DateTimeOriginal', 'GPSLatitude', 'GPSLongitude']
        for tag in relevant:
            if tag in data:
                print(f"  {tag}: {data[tag]}")