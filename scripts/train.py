import tensorflow as tf
from tensorflow.keras import layers
import matplotlib.pyplot as plt
from pathlib import Path

# --- 1. Configuration ---
# Define constants for our data and model. This is a best practice.
DATA_DIR = Path("data/processed")
IMAGE_SIZE = (224, 224) # Standard size for many pre-trained models
BATCH_SIZE = 32 
AUTOTUNE = tf.data.AUTOTUNE # A special value for tf.data to find the best perfromance settings

# --- 2. The Data Pipeline Function ---

def create_data_pipelines():
    """
    Creates and return TensorFlow Fataset objects for train, validation, adn tests sets.
    This function encapsulates all of the data laoding and preprocessing logic.
    """
    # Load the datastets from the directory structre using a Keras utility.
    # It infers class names from the foler names and creates integer labels.
    train_ds = tf.keras.utils.image_dataset_from_directory(
        DATA_DIR / "train",
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        label_mode='int',
        shuffle=True, # shuffle is critical for training
        seed=42 # for reproducibility
    )

    validation_ds = tf.keras.utils.image_dataset_from_directory(
        DATA_DIR / "validation",
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        label_mode='int',
        shuffle=False # no need to shuffle validation data
    )

    test_ds = tf.keras.utils.image_dataset_from_directory(
        DATA_DIR / "test",
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        label_mode='int',
        shuffle=False # no need to shuffle test data
    )

    # Get class names from the training datast. Keras automatically finds them.
    class_names = train_ds.class_names
    print(f"Found {len(class_names)} classes (breeds).")

    # --- 3. Data Augmentation and Preprocessing Layers ---

    # Create a small, sequentail model for our data augmentation
    # These layers will only be applied to the TRAINING data.
    data_augmentation = tf.keras.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.1), # Rotate by a rnadom 10%
        layers.RandomZoom(0.1), # Zoom by a random 10%
    ])

    # Create a layer to normalize pixel values from [0, 255] to [0, 1]
    # This is appled to ALL datasets.
    rescaling = layers.Rescaling(1./255)

    # --- 4. Apply Transformations and Optimize ---

    # Apply the data augmentation ONLY to the training dataset.
    # Use .map() to apply the layers to each batch.
    train_ds = train_ds.map(lambda x,  y: (data_augmentation(x, training=True), y), 
                            num_parallel_calls=AUTOTUNE)
    
    # Apply rescaling to all datasets.
    train_ds = train_ds.map(lambda x, y: (rescaling(x), y), num_parallel_calls=AUTOTUNE)
    validation_ds = validation_ds.map(lambda x, y: (rescaling(x), y), num_parallel_calls=AUTOTUNE)
    test_ds = test_ds.map(lambda x, y: (rescaling(x), y), num_parallel_calls=AUTOTUNE)

    # Configure the datasets for perfromance. This is critical for best practice!
    # .cache() keeps the images in memory after they're loadaded off disk during the first epoch.
    # .prefetch() overlaps data preprocessing and model execution. While the GPU is training 
    # on batch N, the CPU is already preparing batch N+1.
    train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
    validation_ds = validation_ds.cache().prefetch(buffer_size=AUTOTUNE)
    test_ds = test_ds.cache().prefetch(buffer_size=AUTOTUNE)

    return train_ds, validation_ds, test_ds, class_names

# --- 5. Sanity Chcek and Visualization --- 

if __name__ == "__main__":
    # This block will only run when you execute `python scripts/train.py` directly.
    # It's a great way to test our data pipeline.

    train_ds, _, _, class_names = create_data_pipelines()

    print("\n--- Pipeline Sanity Check ---")
    # Use .take(1) to get just one batch from the dataset.
    for images, labels in train_ds.take(1):
        print(f"Image batch shape: {images.shape}")
        print(f"Label batch shape: {labels.shape}")
        
        # Verify that pixel values are now between 0 and 1 
        print(f"Min pixel value: {tf.reduce_min(images[0]).numpy():.4f}")
        print(f"Max pixel value: {tf.reduce_max(images[0]).numpy():.4f}")

        # Visualize the output. This is the best way to confirm the augmentation is working.
        plt.figure(figsize=(10, 10))
        for images, labels in train_ds.take(1):
            for i in range(9):
                ax = plt.subplot(3, 3, i + 1)
                plt.imshow(images[i].numpy())
                plt.title(class_names[labels[i]])
                plt.axis("off")
        plt.suptitle("Sample Augmented Images from the Training Pipeline")
        plt.savefig("data_pipeline_sanity_check.png")
        print("\nSaved a visualization of a sample batch to 'data_pipeline_sanity_check.png'")