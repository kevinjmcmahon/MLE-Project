import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from pathlib import Path
import argparse
import wandb
from wandb.keras import WandbCallback
import os
import hypertune
import numpy as np # Import numpy

# --- 1. Configuration (Defaults) ---
DATA_DIR = Path("data/processed")
IMAGE_SIZE = (300, 300)
BATCH_SIZE = 16 # Keep the safer batch size
AUTOTUNE = tf.data.AUTOTUNE

# --- 2. The Data Pipeline Function ---
def create_data_pipelines():
    """Creates and returns TensorFlow Dataset objects for train, validation, and test sets."""
    train_dir = DATA_DIR / "train"
    num_classes = len(os.listdir(train_dir))
    
    train_ds = tf.keras.utils.image_dataset_from_directory(
        train_dir, image_size=IMAGE_SIZE, batch_size=BATCH_SIZE, label_mode='int', shuffle=True, seed=42
    )
    validation_ds = tf.keras.utils.image_dataset_from_directory(
        DATA_DIR / "validation", image_size=IMAGE_SIZE, batch_size=BATCH_SIZE, label_mode='int', shuffle=False
    )
    class_names = train_ds.class_names
    
    data_augmentation = tf.keras.Sequential([
        layers.RandomFlip("horizontal"), layers.RandomRotation(0.2), layers.RandomZoom(0.2), layers.RandomShear(0.2),
        layers.RandomTranslation(height_factor=0.1, width_factor=0.1), layers.RandomContrast(0.2), layers.RandomBrightness(0.2),
    ])
    
    train_ds = train_ds.map(lambda x, y: (data_augmentation(x, training=True), y), num_parallel_calls=AUTOTUNE)
    train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
    validation_ds = validation_ds.cache().prefetch(buffer_size=AUTOTUNE)
    return train_ds, validation_ds, class_names, num_classes

# --- 3. Model Building Function ---
def build_model(num_classes, model_name="EfficientNetV2B3", dropout_rate=0.5):
    """Builds a transfer learning model with a specified base architecture."""
    if model_name == "EfficientNetV2B3":
        base_model = tf.keras.applications.EfficientNetV2B3(input_shape=(*IMAGE_SIZE, 3), include_top=False, weights='imagenet')
    elif model_name == "EfficientNetV2B2":
         base_model = tf.keras.applications.EfficientNetV2B2(input_shape=(*IMAGE_SIZE, 3), include_top=False, weights='imagenet')
    else:
        raise ValueError(f"Unsupported model name: {model_name}")

    base_model.trainable = False
    inputs = tf.keras.Input(shape=(*IMAGE_SIZE, 3))
    x = base_model(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(dropout_rate)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    model = tf.keras.Model(inputs, outputs)
    return model, base_model

# --- 4. Main Training Function ---
def run_training_pipeline(args):
    """Encapsulates the entire training and logging process."""
    run = wandb.init(project="dog-breed-classification-experiments", config=args, job_type="train")
    
    train_ds, validation_ds, _, num_classes = create_data_pipelines()
    model, base_model = build_model(num_classes, args.model_name, args.dropout_rate)
    
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=args.patience, restore_best_weights=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=1e-7),
        WandbCallback(save_model=False)
    ]
    
    # --- PHASE 1: TRAIN THE HEAD ---
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=args.head_lr),
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                  metrics=['accuracy'])
    history_head = model.fit(train_ds, validation_data=validation_ds, epochs=args.head_epochs, callbacks=callbacks)
    
    # --- PHASE 2: FINE-TUNE THE MODEL ---
    final_history = history_head # Default to head history
    if args.fine_tune_epochs > 0 and args.unfreeze_layers > 0:
        base_model.trainable = True
        for layer in base_model.layers[:-args.unfreeze_layers]:
            layer.trainable = False
        
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=args.fine_tune_lr),
                      loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                      metrics=['accuracy'])
        final_history = model.fit(train_ds, validation_data=validation_ds, epochs=args.fine_tune_epochs, 
                                  initial_epoch=history_head.epoch[-1] if history_head.epoch else 0,
                                  callbacks=callbacks)
    
    # --- FINAL METRIC REPORTING (ROBUST METHOD) ---
    # After all training is done, find the best validation accuracy achieved and report it.
    # This is more robust than reporting on every epoch.
    best_val_accuracy = max(final_history.history['val_accuracy'])
    hpt = hypertune.HyperTune()
    hpt.report_hyperparameter_tuning_metric(
        hyperparameter_metric_tag='val_accuracy',
        metric_value=best_val_accuracy
    )

    # --- SAVE MODEL ---
    model_artifact = wandb.Artifact(name=f"{run.id}-dog-classifier", type="model",
                                    description=f"Model trained with args: {vars(args)}",
                                    metadata=vars(args))
    model_dir = Path("models")
    model_dir.mkdir(exist_ok=True)
    model.save(model_dir / "model.keras")
    model_artifact.add_file(model_dir / "model.keras")
    run.log_artifact(model_artifact)
    run.finish()

# --- Main Execution Block ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a Dog Breed Classifier.")
    parser.add_argument('--head_epochs', type=int, default=15, help='Number of epochs to train the classification head.')
    parser.add_argument('--fine_tune_epochs', type=int, default=30, help='Number of epochs for fine-tuning.')
    parser.add_argument('--patience', type=int, default=10, help='Patience for EarlyStopping.')
    parser.add_argument('--model_name', type=str, default='EfficientNetV2B3', help='Base model architecture to use.')
    parser.add_argument('--dropout_rate', type=float, default=0.5, help='Dropout rate for the classifier head.')
    parser.add_argument('--head_lr', type=float, default=1e-3, help='Learning rate for head training.')
    parser.add_argument('--fine_tune_lr', type=float, default=1e-5, help='Learning rate for fine-tuning.')
    parser.add_argument('--unfreeze_layers', type=int, default=50, help='Number of layers to unfreeze from the end of the base model for fine-tuning.')
    
    args = parser.parse_args()
    run_training_pipeline(args)