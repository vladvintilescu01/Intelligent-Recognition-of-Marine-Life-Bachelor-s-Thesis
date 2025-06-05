from tensorflow import keras
import os

# List of model base names to process
model_bases = [
    "FishImgDataset_augmented_balancedV2_DenseNet121",
    "FishImgDataset_augmented_balancedV2_ResNet50",
    "FishImgDataset_augmented_balancedV2_InceptionV3"
]

# Path to the folder containing the models
base_path = "All weights and structure from experiments"

for base_name in model_bases:
    print(f"\nProcessing model: {base_name}")

    json_path = os.path.join(base_path, f"{base_name}.json")
    weights_path = os.path.join(base_path, f"{base_name}.weights.h5")
    h5_save_path = os.path.join(base_path, f"{base_name}_full.h5")

    # Load model architecture from JSON
    with open(json_path, "r") as f:
        model = keras.models.model_from_json(f.read())

    # Load weights
    model.load_weights(weights_path)

    # Save the complete model as a single .h5 file
    model.save(h5_save_path)
    print(f"Model saved as {h5_save_path}")

print("\nAll models have been processed and saved as .h5 files!")
