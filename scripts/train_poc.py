import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from pathlib import Path

# --- 1. Configuration ---
DATA_DIR = Path("data/processed")
# Updated image size for EfficientNetV2B3 for optimal performance
IMAGE_SIZE = (300, 300) 
BATCH_SIZE = 32 
AUTOTUNE = tf.data.AUTOTUNE
NUM_CLASSES = 120
# Increased epochs to give the model more time to learn
HEAD_EPOCHS = 15
FINE_TUNE_EPOCHS = 50

# --- 2. The Data Pipeline Function ---
def create_data_pipelines():
    """
    Creates and returns TensorFlow Dataset objects for train, validation, and test sets.
    """
    train_ds = tf.keras.utils.image_dataset_from_directory(
        DATA_DIR / "train",
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        label_mode='int',
        shuffle=True,
        seed=42
    )

    validation_ds = tf.keras.utils.image_dataset_from_directory(
        DATA_DIR / "validation",
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        label_mode='int',
        shuffle=False
    )

    test_ds = tf.keras.utils.image_dataset_from_directory(
        DATA_DIR / "test",
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        label_mode='int',
        shuffle=False
    )

    class_names = train_ds.class_names
    print(f"Found {len(class_names)} classes (breeds).")

    # --- Data Augmentation ---
    data_augmentation = tf.keras.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.2),
        layers.RandomZoom(0.2),
        layers.RandomShear(0.2),
        layers.RandomTranslation(height_factor=0.1, width_factor=0.1),
        layers.RandomContrast(0.2),
        layers.RandomBrightness(0.2),
    ])

    # --- Apply Transformations and Optimize ---
    # Apply augmentation only to the training data.
    train_ds = train_ds.map(lambda x, y: (data_augmentation(x, training=True), y), 
                            num_parallel_calls=AUTOTUNE)
    
    # NOTE: EfficientNetV2 models include a rescaling layer. We no longer need our own.
    # The manual rescaling step has been removed from all datasets.

    # Configure datasets for performance.
    train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
    validation_ds = validation_ds.cache().prefetch(buffer_size=AUTOTUNE)
    test_ds = test_ds.cache().prefetch(buffer_size=AUTOTUNE)

    return train_ds, validation_ds, test_ds, class_names

# --- 3. Model Building Function ---
def build_model(num_classes):
    """
    Builds a transfer learning model using the powerful EfficientNetV2B3 architecture.
    """
    # Load the pre-trained EfficientNetV2B3 model
    base_model = tf.keras.applications.EfficientNetV2B3(
        input_shape=(*IMAGE_SIZE, 3),
        include_top=False, # Do not include the final classification layer
        weights='imagenet'
    )

    # Freeze the base model to keep its learned features
    base_model.trainable = False

    # Create the new model by adding a custom classifier head
    inputs = tf.keras.Input(shape=(*IMAGE_SIZE, 3))
    x = base_model(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    # Add a Dropout layer for regularization to prevent overfitting
    x = layers.Dropout(0.5)(x) 
    outputs = layers.Dense(num_classes, activation='softmax')(x)

    model = tf.keras.Model(inputs, outputs)
    return model 

# --- 4. Main Execution Block --- 
if __name__ == "__main__":
    # 1. Create the data pipelines
    train_ds, validation_ds, _, _ = create_data_pipelines()

    # 2. Build the model
    model = build_model(NUM_CLASSES)
    model.summary()

    # Define callbacks that will apply to both training phases
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=1e-6)

    # --- PHASE 1: TRAIN THE HEAD ---
    print(f"\n--- Starting Head Training for {HEAD_EPOCHS} epochs ---")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=['accuracy']
    )

    history_head = model.fit(
        train_ds, 
        validation_data=validation_ds,
        epochs=HEAD_EPOCHS,
        callbacks=[early_stopping, reduce_lr]
    )

    # --- PHASE 2: FINE-TUNE THE MODEL ---
    print(f"\n--- Starting Fine-Tuning for {FINE_TUNE_EPOCHS} epochs ---")
    base_model = model.layers[1] # Get the base model layer
    base_model.trainable = True # Unfreeze the entire base model

    # Recompile the model with a very low learning rate for fine-tuning
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=['accuracy']
    )   

    # Continue training the model
    history_fine_tune = model.fit(
        train_ds, 
        validation_data=validation_ds,
        epochs=FINE_TUNE_EPOCHS,
        callbacks=[early_stopping, reduce_lr]
    )

    # 5. Save the final trained model
    print("\n--- Training Complete ---")
    Path("models").mkdir(exist_ok=True)
    # Update the model name to reflect the new architecture
    model.save("models/efficientnetv2b3_dog_classifier.keras")
    print("Model saved to 'models/efficientnetv2b3_dog_classifier.keras'")