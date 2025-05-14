import os
import shutil
import random
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img, array_to_img
from tqdm import tqdm

# Paths
original_dataset_path = 'D:/Facultate_ACE/Facultate_Anul_IV/ML/FishImgDataset/train'
balanced_dataset_path = 'D:/Facultate_ACE/Facultate_Anul_IV/ML/FishImgDataset_augmented_balanced/train'
# Target number of images per class
maximum_images_per_classes = 1000
# Augmentation settings
datagen = ImageDataGenerator(
    rotation_range = 4,
    width_shift_range = 0.05,
    height_shift_range = 0.05,
    zoom_range = 0.04,
    horizontal_flip = True,
    brightness_range = [0.5, 1.2],
    fill_mode = 'nearest'
)
# Get all class names (folder names)
categories = [category for category in os.listdir(original_dataset_path) if os.path.isdir(os.path.join(original_dataset_path, category))]
# Process each class
for categories in tqdm(categories, desc="I am working at this dataset..."):
    src_dir = os.path.join(original_dataset_path, categories)
    dest_dir = os.path.join(balanced_dataset_path, categories)
    os.makedirs(dest_dir)
    # Get all image files in the class folder
    all_images = [image for image in os.listdir(src_dir) if image.lower().endswith(('.png', '.jpg', '.jpeg'))]
    current_images_per_classes = len(all_images)
    if current_images_per_classes < maximum_images_per_classes:
        # Copy all available images
        for image in all_images:
            shutil.copy2(os.path.join(src_dir, image), os.path.join(dest_dir, image))
        # Augment more if we need extra images
        new_images_augmented = maximum_images_per_classes - current_images_per_classes
        if new_images_augmented > 0:
            print(f"\nI will augument with {new_images_augmented} images this category: '{categories}'...")
            while new_images_augmented > 0:
                # Pick a random image from the original class
                image = random.choice(all_images)
                image_path = os.path.join(src_dir, image)
                # Load and convert to array
                image_loaded = load_img(image_path)
                image_transformed_in_array = img_to_array(image_loaded)
                image_transformed_in_array = image_transformed_in_array.reshape((1,) + image_transformed_in_array.shape)
                # Start augmenting
                augumentation = datagen.flow(image_transformed_in_array, batch_size=1)
                count_of_augumentation = 100
                while count_of_augumentation > 0 and new_images_augmented > 0:
                    augumented_image = next(augumentation)[0]
                    save_path = os.path.join(dest_dir, f"augmented_image_{random.randint(1000, 999999)}.jpg")
                    array_to_img(augumented_image).save(save_path)
                    count_of_augumentation -= 1
                    new_images_augmented -= 1
                    if new_images_augmented == 0:
                        break


print("\nThe new dataset created successfully at:", balanced_dataset_path)