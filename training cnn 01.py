import tensorflow as tf
from tensorflow.keras.applications import EfficientNetV2S
from tensorflow.keras.applications.efficientnet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models, optimizers, callbacks
import os

# Configuration (High Accuracy Settings)
DATASET_DIR = "Potato_Disease_Classification"
IMG_SIZE = (224, 224)
BATCH_SIZE = 16        # smaller batch → better generalization
EPOCHS = 60            # fine-tuning included
NUM_CLASSES = 4

# Data Generators (Disease-Safe Augmentation)
train_gen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    rotation_range=15,
    width_shift_range=0.05,
    height_shift_range=0.05,
    zoom_range=0.15,
    brightness_range=[0.85, 1.15],
    horizontal_flip=True
)

val_gen = ImageDataGenerator(
    preprocessing_function=preprocess_input
)

train_data = train_gen.flow_from_directory(
    os.path.join(DATASET_DIR, "train"),
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical"
)

val_data = val_gen.flow_from_directory(
    os.path.join(DATASET_DIR, "val"),
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    shuffle=False
)

# Build EfficientNetV2 Model
base_model = EfficientNetV2S(
    include_top=False,
    weights="imagenet",
    input_shape=(224, 224, 3)
)

base_model.trainable = False   # 🔒 Stage-1: freeze backbone

# Custom Classification Head (Accuracy-Focused)
x = base_model.output
x = layers.GlobalAveragePooling2D()(x)
x = layers.BatchNormalization()(x)
x = layers.Dense(512, activation="relu")(x)
x = layers.Dropout(0.4)(x)
output = layers.Dense(NUM_CLASSES, activation="softmax")(x)

model = models.Model(inputs=base_model.input, outputs=output)

# Compile (Stable & Accurate)
model.compile(
    optimizer=optimizers.AdamW(learning_rate=1e-3, weight_decay=1e-4),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

# Callbacks (Critical for High Accuracy)
cb = [
    callbacks.EarlyStopping(patience=10, restore_best_weights=True),
    callbacks.ReduceLROnPlateau(factor=0.3, patience=5, min_lr=1e-6),
    callbacks.ModelCheckpoint(
        "EfficientNetV2_PotatoDisease.h5",
        monitor="val_accuracy",
        save_best_only=True
    )
]

# Stage-1 Training (Classifier Head Only)
history_1 = model.fit(
    train_data,
    validation_data=val_data,
    epochs=20,
    callbacks=cb
)

# Stage-2 Fine-Tuning (VERY IMPORTANT)
base_model.trainable = True

for layer in base_model.layers[:-40]:   # freeze early layers
    layer.trainable = False

model.compile(
    optimizer=optimizers.AdamW(learning_rate=1e-4, weight_decay=1e-5),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

history_2 = model.fit(
    train_data,
    validation_data=val_data,
    epochs=EPOCHS,
    callbacks=cb
)