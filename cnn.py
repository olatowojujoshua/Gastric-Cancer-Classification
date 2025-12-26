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


AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.prefetch(buffer_size=AUTOTUNE)

data_augmentation = keras.Sequential(
    [
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.1),
        layers.RandomZoom(0.1),
    ]
)

normalization_layer = layers.Rescaling(1./225)

#custom model
from tensorflow import keras
from tensorflow.keras import layers

normalization_layer = layers.Rescaling(1./225)

inputs = keras.Input(shape=img_size + (3,))

x = normalization_layer(inputs)

# Block 1
x = layers.Conv2D(32, (3,3), activation="relu", padding="same")(x)
x = layers.MaxPooling2D((2,2))(x)

# Block 2
x = layers.Conv2D(64, (3,3), activation="relu", padding="same")(x)
x = layers.MaxPooling2D((2,2))(x)

# Block 3
x = layers.Conv2D(128, (3,3), activation="relu", padding="same")(x)
x = layers.MaxPooling2D((2,2))(x)

# Global pooling + classifier
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dropout(0.3)(x)
outputs = layers.Dense(num_classes, activation="softmax")(x)

model = keras.Model(inputs, outputs)

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-3),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

