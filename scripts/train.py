import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from pathlib import Path
import argparse
import wandb  # W&B: Import the Weights & Biases library
from wandb.keras import WandbCallback  # W&B: Import the Keras callback
import os  # VERTEX AI CHANGE: Import the 'os' library to dynamically count classes
import hypertune # VERTEX AI CHANGE: Import the 'hypertune' library to report metrics

# --- 1. Configuration (Defaults) ---
# These are now just default values. They will be overridden by command-line args.
DATA_DIR = Path("data/processed")
IMAGE_SIZE = (300, 300)
BATCH_SIZE = 16
AUTOTUNE = tf.data.AUTOTUNE

# --- 2. The Data Pipeline Function (Modified for dynamic class counting) ---
def create_data_pipelines():
    """Creates and returns TensorFlow Dataset objects for train, validation, and test sets."""
    train_dir = DATA_DIR / "train"  # Define the training directory path

    # VERTEX AI CHANGE: Dynamically determine the number of classes from the directory structure.
    # This is much more robust than hardcoding the number of classes.
    num_classes = len(os.listdir(train_dir))
    print(f"Dynamically found {num_classes} classes (breeds).")
    
    # (Your data pipeline code is excellent and remains unchanged)
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
    
    # VERTEX AI CHANGE: Return the dynamically found num_classes so the model can use it.
    return train_ds, validation_ds, class_names, num_classes

# --- 3. Model Building Function (No changes needed) ---
def build_model(num_classes, model_name="EfficientNetV2B3", dropout_rate=0.5):
    """Builds a transfer learning model with a specified base architecture."""
    # Parameterizing the model name and dropout allows for easy experimentation.
    if model_name == "EfficientNetV2B3":
        base_model = tf.keras.applications.EfficientNetV2B3(
            input_shape=(*IMAGE_SIZE, 3), include_top=False, weights='imagenet'
        )
    elif model_name == "EfficientNetV2B2":
         base_model = tf.keras.applications.EfficientNetV2B2(
            input_shape=(*IMAGE_SIZE, 3), include_top=False, weights='imagenet'
        )
    # Add other models here if you want to experiment further (e.g., ResNet, Inception)
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

# --- 4. Main Training Function (Modified for dynamic class counting and HP Tuning) ---
def run_training_pipeline(args):
    """Encapsulates the entire training and logging process."""
    
    # W&B: Initialize a new experiment run.
    # This logs all hyperparameters from `args` to the W&B dashboard.
    run = wandb.init(
        project="dog-breed-classification-experiments",  # Group runs under this project
        config=args,
        job_type="train"
    )

    # 1. Create the data pipelines and get the dynamic class count
    train_ds, validation_ds, _, num_classes = create_data_pipelines()

    # 2. Build the model using the dynamic class count
    model, base_model = build_model(num_classes, args.model_name, args.dropout_rate)
    
    # VERTEX AI CHANGE: Define a custom Keras callback to report metrics to Vertex AI.
    class HypertuneCallback(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            hpt = hypertune.HyperTune()
            hpt.report_hyperparameter_tuning_metric(
                hyperparameter_metric_tag='val_accuracy',  # This tag MUST match the one in your config.yaml
                metric_value=logs['val_accuracy'],
                global_step=epoch
            )

    # W&B: Define all callbacks, including the WandbCallback.
    # WandbCallback will automatically log metrics, losses, and learning rate.
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=args.patience, restore_best_weights=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=1e-7),
        WandbCallback(save_model=False), # We will save the model as a W&B Artifact instead.
        HypertuneCallback() # VERTEX AI CHANGE: Add the Hypertune callback.
    ]

    # --- PHASE 1: TRAIN THE HEAD ---
    print(f"\n--- Starting Head Training for {args.head_epochs} epochs with LR={args.head_lr} ---")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=args.head_lr),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=['accuracy']
    )
    history_head = model.fit(
        train_ds, validation_data=validation_ds, epochs=args.head_epochs, callbacks=callbacks
    )

    # --- PHASE 2: FINE-TUNE THE MODEL ---
    # PRO-TIP: This block allows you to run a "feature-extraction-only" experiment
    # by setting fine_tune_epochs=0 or unfreeze_layers=0.
    if args.fine_tune_epochs > 0 and args.unfreeze_layers > 0:
        print(f"\n--- Starting Fine-Tuning for {args.fine_tune_epochs} epochs with LR={args.fine_tune_lr} ---")
        
        base_model.trainable = True
        print(f"Unfreezing the last {args.unfreeze_layers} layers of the base model.")
        for layer in base_model.layers[:-args.unfreeze_layers]:
            layer.trainable = False

        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=args.fine_tune_lr),
            loss=tf.keras.losses.SparseCategoricalCrossentropy(),
            metrics=['accuracy']
        )
        model.fit(
            train_ds, validation_data=validation_ds, epochs=args.fine_tune_epochs, 
            initial_epoch=history_head.epoch[-1] if history_head.epoch else 0, # Continue epoch count
            callbacks=callbacks
        )

    # --- 5. Save the final model as a W&B Artifact ---
    print("\n--- Training Complete, logging model artifact to W&B ---")
    
    # W&B: This is the professional way to save and version your model.
    # It links the model file directly to the run that produced it.
    model_artifact = wandb.Artifact(
        name=f"{run.id}-dog-classifier", # Give the artifact a unique name
        type="model",
        description=f"Model trained with args: {vars(args)}",
        metadata=vars(args)
    )
    
    model_dir = Path("models")
    model_dir.mkdir(exist_ok=True)
    model.save(model_dir / "model.keras")
    model_artifact.add_file(model_dir / "model.keras") # Add the saved model file to the artifact
    run.log_artifact(model_artifact) # Log the artifact to the W&B run

    # W&B: End the current run
    run.finish()

# --- Main Execution Block ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a Dog Breed Classifier.")
    # Training process arguments
    parser.add_argument('--head_epochs', type=int, default=15, help='Number of epochs to train the classification head.')
    parser.add_argument('--fine_tune_epochs', type=int, default=30, help='Number of epochs for fine-tuning.')
    parser.add_argument('--patience', type=int, default=10, help='Patience for EarlyStopping.')
    # Model architecture arguments
    parser.add_argument('--model_name', type=str, default='EfficientNetV2B3', help='Base model architecture to use.')
    parser.add_argument('--dropout_rate', type=float, default=0.5, help='Dropout rate for the classifier head.')
    # Hyperparameter arguments
    parser.add_argument('--head_lr', type=float, default=1e-3, help='Learning rate for head training.')
    parser.add_argument('--fine_tune_lr', type=float, default=1e-5, help='Learning rate for fine-tuning.')
    parser.add_argument('--unfreeze_layers', type=int, default=50, help='Number of layers to unfreeze from the end of the base model for fine-tuning. Set to 0 for feature-extraction only.')
    
    args = parser.parse_args()
    run_training_pipeline(args)