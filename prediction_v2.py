import os
import cv2
import shutil
import numpy as np
from keras.models import model_from_json, load_model

# Define paths
base_path = "D:/Facultate_ACE/Facultate_Anul_IV/ML/"
# Choose one of the datasets:
# Fish_Dataset_Split / FishImgDataset / FishImgDataset–modified / FishImgDataset_augmented_balanced /FishImgDataset_augmented_balancedV2
test_folder = os.path.join(base_path, "FishImgDataset_augmented_balancedV2/test") 

# Load model architecture using JSON and H5
# Choose one of the datasets:
# Fish_Dataset_Split / FishImgDataset / FishImgDataset–modified / FishImgDataset_augmented_balanced / FishImgDataset_augmented_balancedV2
# Choose one of the models:
# Used for: _ResNet50.json / _DenseNet121.json / _InceptionV3.json
with open(base_path + 'FishImgDataset_augmented_balancedV2_InceptionV3.json', 'r') as json_file:
    loaded_model_json = json_file.read()

model = model_from_json(loaded_model_json)
# Load model architecture using JSON and H5
# Choose one of the datasets:
# Fish_Dataset_Split / FishImgDataset / FishImgDataset–modified / FishImgDataset_augmented_balanced / FishImgDataset_augmented_balancedV2
# Choose one of the models:
# _ResNet50.weights.h5 / _DenseNet121.weights.h5 / _InceptionV3.weights.h5
model.load_weights(base_path + "FishImgDataset_augmented_balancedV2_InceptionV3.weights.h5") 

# First dataset, a more laborious environment
# CLASSES = np.array(["Black Sea Sprat", "Gilt-Head Bream", "Hourse Mackerel", "Red Mullet",
#                    "Red Sea Bream", "Sea Bass", "Shrimp", "Striped Red Mullet"])  

# Second dataset, more diversity
CLASSES = np.array(["Catfish", "Glass Perchlet", "Goby", "Gourami", 
                    "Grass Carp", "Knifefish", "Silver Barb", "Tilapia"])

# Preprocess image
def preprocess_image(img_path):
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (224, 224))
    img = img.astype('float32') / 255.0
    img = np.expand_dims(img, axis=0)
    return img

# Counters
correct = 0
incorrect = 0
misclassified = []

# Predict all images in the test folder
for class_dir in os.listdir(test_folder):
    class_path = os.path.join(test_folder, class_dir)
    if not os.path.isdir(class_path):
        continue
    for image_name in os.listdir(class_path):
        image_path = os.path.join(class_path, image_name)
        try:
            img = preprocess_image(image_path)
            pred = model.predict(img)
            predicted_class = CLASSES[np.argmax(pred)]
            confidence = np.max(pred)

            if predicted_class == class_dir:
                correct += 1
            else:
                incorrect += 1
                misclassified.append({
                    "image": image_name,
                    "actual": class_dir,
                    "predicted": predicted_class,
                    "confidence": confidence
                })

        except Exception as e:
            print(f"Error with image {image_name}: {e}")

# Calculate accuracy
total_images = correct + incorrect
accuracy = (correct/total_images) * 100

# Results summary
print("\n--- Prediction Summary ---")
print(f"Correct predictions: {correct}")
print(f"Incorrect predictions: {incorrect}")
print(f"Accuracy on this test: {accuracy:.1f}%")

# Show incorrect predictions
if misclassified:
    print("\n--- Misclassified Images ---")
    for item in misclassified:
        print(f"Image: {item['image']}")
        print(f"Actual class: {item['actual']}")
        print(f"Predicted class: {item['predicted']} (Confidence: {item['confidence']:.2f})")
        print("-" * 50)
else:
    print("All images were predicted correctly")
