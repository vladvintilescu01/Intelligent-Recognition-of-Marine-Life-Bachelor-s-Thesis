# -*- coding: utf-8 -*-
import tensorflow as tf
import os
import random
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Model
from tensorflow.keras.layers import GlobalAveragePooling2D, Flatten, Dense, Dropout, BatchNormalization, LeakyReLU, PReLU
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers.experimental import AdamW
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.metrics import classification_report, confusion_matrix


# GPU configuration
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(len(gpus), "Physical GPUs,", len(tf.config.list_logical_devices('GPU')), "Logical GPUs")
    except RuntimeError as e:
        print(e)

with tf.device('/GPU:0'):
    # Reproducibility
    seed = 42
    np.random.seed(seed)
    random.seed(seed)
    tf.random.set_seed(seed)
    
    # Image and path settings
    IMG_HEIGHT, IMG_WIDTH, CHANNELS = 224, 224, 3
    BATCH_SIZE = 16
    EPOCHS = 20 
    # Choose one of the datasets:
    # Fish_Dataset_Split / FishImgDataset / FishImgDataset-modified / FishImgDataset_augmented_balanced / FishImgDataset_augmented_balancedV2 / FishImgDataset_18_classes_augmented_balancedV2
    DATA_PATH = 'D:/Facultate_ACE/Facultate_Anul_IV/ML/FishImgDataset_18_classes_augmented_balancedV2' 
    SAVE_PATH = 'D:/Facultate_ACE/Facultate_Anul_IV/ML/'
    
    #Early Stopping, when model stops to get better loss
    early_stop = EarlyStopping(
    monitor='val_loss',
    patience=3,
    restore_best_weights=True
    )
    
    # ImageDataGenerator setup without augmentation, but using manual split for train & validation
    train_gen = ImageDataGenerator(rescale=1./255).flow_from_directory(
        os.path.join(DATA_PATH, 'train'),
        target_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        shuffle=True,
        seed=seed
    )

    val_gen = ImageDataGenerator(rescale=1./255).flow_from_directory(
        os.path.join(DATA_PATH, 'val'),
        target_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        shuffle=False,
        seed=seed
    )

    # ImageDataGenerator setup with augmentation, but using manual split for train & validation
    # train_datagen = ImageDataGenerator(
    #     rescale=1./255,
    #     rotation_range = 4,
    #     width_shift_range = 0.05,
    #     height_shift_range = 0.05,
    #     zoom_range = 0.04,
    #     horizontal_flip = True,
    #     brightness_range = [0.5, 1.2],
    #     fill_mode = 'nearest'
    # )

    # train_gen = train_datagen.flow_from_directory(
    #     os.path.join(DATA_PATH, 'train'),
    #     target_size=(IMG_HEIGHT, IMG_WIDTH),
    #     batch_size=BATCH_SIZE,
    #     class_mode='categorical',
    #     shuffle=True,
    #     seed=seed
    # )

    # val_gen = ImageDataGenerator(rescale=1./255).flow_from_directory(
    #     os.path.join(DATA_PATH, 'val'),
    #     target_size=(IMG_HEIGHT, IMG_WIDTH),
    #     batch_size=BATCH_SIZE,
    #     class_mode='categorical',
    #     shuffle=False,
    #     seed=seed
    # )
        
    NUM_CLASSES = train_gen.num_classes
    
    # Model definition
    def build_resnet50_model(input_shape=(224, 224, 3), num_classes=NUM_CLASSES, 
                             layer1=256, layer2=128, dropout_rate=0.3, lr=0.00001):
        base_model = ResNet50(weights = 'imagenet', include_top=False, input_shape=(224, 224, 3))
        #mark loaded layers as trainable
        for layer in base_model.layers[-30:]:
          layer.trainable = True
        
        x = GlobalAveragePooling2D()(base_model.output)
        
        x = Dense(layer1)(x)
        x = BatchNormalization()(x)
        x = PReLU()(x)
        x = Dropout(dropout_rate)(x)
        
        x = Dense(layer2)(x)
        x = BatchNormalization()(x)
        x = PReLU()(x)
        x = Dropout(dropout_rate)(x)
        
        output = Dense(num_classes, activation='softmax')(x)
    
        model = Model(inputs=base_model.input, outputs=output)
        model.compile(loss='categorical_crossentropy',
                      optimizer=Adam(learning_rate=lr),
                      metrics=['accuracy'])
        return model
    
    model = build_resnet50_model()
    model.summary()
    
    #Train the model without augmentation
    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=EPOCHS,
        callbacks=[early_stop]
    )
    
    # Train the model with augmentation
    # steps_per_epoch = None
    
    # history = model.fit(
    #     train_gen,
    #     steps_per_epoch=steps_per_epoch,
    #     validation_data=val_gen,
    #     validation_steps=None,
    #     epochs=EPOCHS,
    #     callbacks=[early_stop]
    # )

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train')
    plt.plot(history.history['val_accuracy'], label='Val')
    plt.title('Accuracy')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train')
    plt.plot(history.history['val_loss'], label='Val')
    plt.title('Loss')
    plt.legend()
    
    plt.show()

    # Save the model
    # Choose one of the datasets:
    # Fish_Dataset_Split / FishImgDataset / FishImgDataset–modified / FishImgDataset_augmented_balanced / FishImgDataset_augmented_balancedV2 / FishImgDataset_18_classes_augmented_balancedV2
    model_json = model.to_json()
    with open(SAVE_PATH + 'FishImgDataset_18_classes_augmented_balancedV2_ResNet50.json', 'w') as json_file:
        json_file.write(model_json)
    model.save_weights(SAVE_PATH + 'FishImgDataset_18_classes_augmented_balancedV2_ResNet50.weights.h5')
    print("Model saved to disk")
    
    # Evaluate the model and print classification report for validation set
    # Prediction on validation
    y_pred = model.predict(val_gen, verbose=1)

    # True label
    y_true = val_gen.classes

    # Get predicted classes
    y_pred_classes = np.argmax(y_pred, axis=1)

    print("\nClassification Report:")
    print(classification_report(y_true, y_pred_classes, target_names=val_gen.class_indices.keys()))

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred_classes)
    
    plt.figure(figsize=(10, 8)) 
    sns.heatmap(cm.astype("float") / cm.sum(axis=1)[:, np.newaxis],
                annot=True, fmt=".2f", cmap="Blues", 
                xticklabels=val_gen.class_indices.keys(), 
                yticklabels=val_gen.class_indices.keys())
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.show()

