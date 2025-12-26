import kagglehub

# Download latest version
path = kagglehub.dataset_download("andrewmvd/gastric-cancer-classification")

print("Path to dataset files:", path)

import os

dataset_path = "/root/.cache/kagglehub/datasets/andrewmvd/gastric-cancer-classification/versions/2/Images"

for root, dirs, files in os.walk(dataset_path):
    print(root, len(files))

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

data_dir = "/root/.cache/kagglehub/datasets/andrewmvd/gastric-cancer-classification/versions/2/Images/Images"

img_size = (100, 100)
batch_size = 32
seed = 42

# Train / validation split from folders
train_ds = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="training",
    seed=seed,
    image_size=img_size,
    batch_size=batch_size
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="validation",
    seed=seed,
    image_size=img_size,
    batch_size=batch_size
)

class_names = train_ds.class_names
num_classes = len(class_names)
print("Classes:", class_names)
