import tensorflow as tf
import keras_cv
import pandas as pd
import numpy as np
from tensorflow import keras
import os

# ---------- CONFIGURATION ----------
IMAGE_SIZE = (640, 640)
BATCH_SIZE = 2          # reduce to 1 if you get out-of-memory errors
EPOCHS = 30
NUM_CLASSES = 15      # number of disease classes (excluding Invalid)

# Replace with your actual class names (exactly as typed in LabelImg)
CLASS_NAMES = [
    'Corn___Common_Rust',
    'Corn___Healthy',
    'Corn___Leaf_Blight',
    'Potato___Early_Blight',
    'Potato___Healthy',
    'Potato___Late_Blight',
    'Rice___Brown_Spot',
    'Rice___Healthy',
    'Rice___Hispa',
    'Rice___Leaf_Blast',
    'Wheat___Brown_Rust',
    'Wheat___Healthy',
    'Wheat___Yellow_Rust'
]
# -----------------------------------

df = pd.read_csv('detection_data/annotations/train_labels.csv')
# Split 80% train, 20% validation
train_df = df.sample(frac=0.8, random_state=42)
val_df = df.drop(train_df.index)

def build_dataset(dataframe, image_dir, is_training=True):
    grouped = dataframe.groupby('filename')
    images, boxes_list, classes_list = [], [], []
    for fname, group in grouped:
        img_path = os.path.join(image_dir, fname)
        img = tf.io.read_file(img_path)
        img = tf.image.decode_jpeg(img, channels=3)
        img = tf.image.resize(img, IMAGE_SIZE)
        img = tf.cast(img, tf.float32) / 255.0
        boxes, classes = [], []
        for _, row in group.iterrows():
            xmin = row['xmin'] / row['width']
            ymin = row['ymin'] / row['height']
            xmax = row['xmax'] / row['width']
            ymax = row['ymax'] / row['height']
            boxes.append([xmin, ymin, xmax, ymax])
            class_idx = CLASS_NAMES.index(row['class']) + 1   # 1‑based
            classes.append(class_idx)
        images.append(img)
        boxes_list.append(boxes)
        classes_list.append(classes)
    dataset = tf.data.Dataset.from_tensor_slices((images, boxes_list, classes_list))
    if is_training:
        dataset = dataset.shuffle(100).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    else:
        dataset = dataset.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    return dataset

train_ds = build_dataset(train_df, 'detection_data/images', is_training=True)
val_ds = build_dataset(val_df, 'detection_data/images', is_training=False)

model = keras_cv.models.FasterRCNN(
    num_classes=NUM_CLASSES,
    bounding_box_format="xyxy",
    backbone="resnet50",
    include_rescaling=False
)

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    classification_loss="categorical_crossentropy",
    box_loss="huber"
)

print("Starting training...")
history = model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS)

print("Evaluating...")
model.evaluate(val_ds)
model.save('faster_rcnn_model.keras')
print("Model saved.")