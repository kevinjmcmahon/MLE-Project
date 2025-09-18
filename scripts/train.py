import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from pathlib import Path
import argparse
import wandb  # W&B: Import the Weights & Biases library
from wandb.keras import WandbCallback  # W&B: Import the Keras callback
import os
import hypertune # This is for the hypertune library to report metrics

# --- 1. Configuration (Defaults) ---
# These are now just default values. They will be overridden by command-line args.
DATA_DIR = Path("data/processed")
IMAGE_SIZE = (300, 300)
BATCH_SIZE = 16 # Kept at 16 from our previous debugging step
AUTOTUNE = tf.data.AUTOTUNE

# --- 2. The Data Pipeline Function (with verbose debugging) ---
def create_data_pipelines():
    """Creates and returns TensorFlow Dataset objects for train, validation, and test sets."""
    print("--- [DEBUG] Entered create_data_pipelines function.") # <-- ADDED
    train_dir = DATA_DIR / "train"

    num_classes = len(os.listdir(train_dir))
    print(f"--- [DEBUG] Dynamically found {num_classes} classes (breeds).") # <-- ADDED

    train_ds = tf.keras.utils.image_dataset_from_directory(
        train_dir, image_size=IMAGE_SIZE, batch_size=BATCH_SIZE, label_mode='int', shuffle=True, seed=42
    )
    print("--- [DEBUG] image_dataset_from_directory for train_ds FINISHED.") # <-- ADDED

    validation_ds = tf.keras.utils.image_dataset_from_directory(
        DATA_DIR / "validation", image_size=IMAGE_SIZE, batch_size=BATCH_SIZE, label_mode='int', shuffle=False
    )
    print("--- [DEBUG] image_dataset_from_directory for validation_ds FINISHED.") # <-- ADDED

    class_names = train_ds.class_names
    
    data_augmentation = tf.keras.Sequential([
        layers.RandomFlip("horizontal"), layers.RandomRotation(0.2), layers.RandomZoom(0.2), layers.RandomShear(0.2),
        layers.RandomTranslation(height_factor=0.1, width_factor=0.1), layers.RandomContrast(0.2), layers.RandomBrightness(0.2),
    ])
    
    print("--- [DEBUG] Mapping data augmentation.") # <-- ADDED
    train_ds = train_ds.map(lambda x, y: (data_augmentation(x, training=True), y), num_parallel_calls=AUTOTUNE)
    print("--- [DEBUG] Data augmentation mapping FINISHED.") # <-- ADDED

    print("--- [DEBUG] Caching and prefetching datasets.") # <-- ADDED
    train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
    validation_ds = validation_ds.cache().prefetch(buffer_size=AUTOTUNE)
    print("--- [DEBUG] Caching and prefetching FINISHED.") # <-- ADDED

    print("--- [DEBUG] create_data_pipelines function FINISHED.") # <-- ADDED
    return train_ds, validation_ds, class_names, num_classes

# --- 3. Model Building Function ---
def build_model(num_classes, model_name="EfficientNetV2B3", dropout_rate=0.5):
    """Builds a transfer learning model with a specified base architecture."""
    if model_name == "EfficientNetV2B3":
        base_model = tf.keras.applications.EfficientNetV2B3(
            input_shape=(*IMAGE_SIZE, 3), include_top=False, weights='imagenet'
        )
    elif model_name == "EfficientNetV2B2":
         base_model = tf.keras.applications.EfficientNetV2B2(
            input_shape=(*IMAGE_SIZE, 3), include_top=False, weights='imagenet'
        )
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

# --- 4. Main Training Function (with verbose debugging) ---
def run_training_pipeline(args):
    """Encapsulates the entire training and logging process."""
    print("--- [DEBUG] Entered run_training_pipeline function.") # <-- ADDED
    
    run = wandb.init(
        project="dog-breed-classification-experiments",
        config=args,
        job_type="train"
    )

    # 1. Create the data pipelines
    print("--- [DEBUG] Calling create_data_pipelines.") # <-- ADDED
    train_ds, validation_ds, _, num_classes = create_data_pipelines()
    print("--- [DEBUG] create_data_pipelines has returned.") # <-- ADDED

    # 2. Build the model
    print("--- [DEBUG] Building the model.") # <-- ADDED
    model, base_model = build_model(num_classes, args.model_name, args.dropout_rate)
    print("--- [DEBUG] Model build complete.") # <-- ADDED

    # Define a custom Keras callback for Vertex AI Hypertune.
    class HypertuneCallback(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            hpt = hypertune.HyperTune()
            hpt.report_hyperparameter_tuning_metric(
                hyperparameter_metric_tag='val_accuracy',
                metric_value=logs['val_accuracy'],
                global_step=epoch
            )

    callbacks = [
        EarlyStopping(monitor='val_loss', patience=args.patience, restore_best_weights=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=1e-7),
        WandbCallback(save_model=False),
        HypertuneCallback()
    ]

    # --- PHASE 1: TRAIN THE HEAD ---
    print(f"\n--- [DEBUG] Compiling model for head training.") # <-- ADDED
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=args.head_lr),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=['accuracy']
    )
    print(f"--- [DEBUG] Starting model.fit for head training.") # <-- ADDED
    history_head = model.fit(
        train_ds, validation_data=validation_ds, epochs=args.head_epochs, callbacks=callbacks
    )
    print(f"--- [DEBUG] model.fit for head training FINISHED.") # <-- ADDED

    # --- PHASE 2: FINE-TUNE THE MODEL ---
    if args.fine_tune_epochs > 0 and args.unfreeze_layers > 0:
        print(f"\n--- [DEBUG] Starting Fine-Tuning phase.") # <-- ADDED
        
        base_model.trainable = True
        print(f"--- [DEBUG] Unfreezing the last {args.unfreeze_layers} layers of the base model.") # <-- ADDED
        for layer in base_model.layers[:-args.unfreeze_layers]:
            layer.trainable = False

        print(f"--- [DEBUG] Compiling model for fine-tuning.") # <-- ADDED
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=args.fine_tune_lr),
            loss=tf.keras.losses.SparseCategoricalCrossentropy(),
            metrics=['accuracy']
        )
        print(f"--- [DEBUG] Starting model.fit for fine-tuning.") # <-- ADDED
        model.fit(
            train_ds, validation_data=validation_ds, epochs=args.fine_tune_epochs, 
            initial_epoch=history_head.epoch[-1] if history_head.epoch else 0,
            callbacks=callbacks
        )
        print(f"--- [DEBUG] model.fit for fine-tuning FINISHED.") # <-- ADDED

    # --- 5. Save the final model as a W&B Artifact ---
    print("\n--- [DEBUG] Training Complete, logging model artifact to W&B ---") # <-- ADDED
    
    model_artifact = wandb.Artifact(
        name=f"{run.id}-dog-classifier",
        type="model",
        description=f"Model trained with args: {vars(args)}",
        metadata=vars(args)
    )
    
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
    parser.add_gument('--patience', type=int, default=10, help='Patience for EarlyStopping.')
    parser.add_argument('--model_name', type=str, default='EfficientNetV2B3', help='Base model architecture to use.')
    parser.add_argument('--dropout_rate', type=float, default=0.5, help='Dropout rate for the classifier head.')
    parser.add_argument('--head_lr', type=float, default=1e-3, help='Learning rate for head training.')
    parser.add_argument('--fine_tune_lr', type=float, default=1e-5, help='Learning rate for fine-tuning.')
    parser.add_argument('--unfreeze_layers', type=int, default=50, help='Number of layers to unfreeze from the end of the base model for fine-tuning. Set to 0 for feature-extraction only.')
    
    args = parser.parse_args()
    run_training_pipeline(args)