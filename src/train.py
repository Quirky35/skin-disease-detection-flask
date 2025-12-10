import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.utils.class_weight import compute_class_weight

# Paths
base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
train_dir = os.path.join(base_dir, 'dataset', 'train')
test_dir = os.path.join(base_dir, 'dataset', 'test')
model_path = os.path.join(base_dir, 'models', 'skin_disease_model.h5')
os.makedirs(os.path.dirname(model_path), exist_ok=True)

# Parameters
IMAGE_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 20

# Step 1: Compute class counts
class_names = sorted([d for d in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, d))])
class_counts = {}

for idx, cls in enumerate(class_names):
    cls_folder = os.path.join(train_dir, cls)
    image_count = len([f for f in os.listdir(cls_folder) if os.path.isfile(os.path.join(cls_folder, f))])
    class_counts[idx] = image_count

print("✅ Class counts:", class_counts)

# Step 2: Compute class weights
labels = []
for class_idx, count in class_counts.items():
    labels += [class_idx] * count

class_weights_array = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(labels),
    y=labels
)
class_weights = {i: float(w) for i, w in enumerate(class_weights_array)}
print("✅ Computed class weights:", class_weights)

# Step 3: Data generators
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    zoom_range=0.2,
    horizontal_flip=True
)

test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(IMAGE_SIZE, IMAGE_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(IMAGE_SIZE, IMAGE_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)

# Step 4: Load base model and add custom layers
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3))

# Fine-tune last 50 layers
for layer in base_model.layers[:-50]:
    layer.trainable = False
for layer in base_model.layers[-50:]:
    layer.trainable = True

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(128, activation='relu')(x)
predictions = Dense(train_generator.num_classes, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=predictions)

# Step 5: Compile model
model.compile(optimizer=Adam(learning_rate=1e-4), loss='categorical_crossentropy', metrics=['accuracy'])

# Step 6: Callbacks
checkpoint = ModelCheckpoint(model_path, monitor='val_accuracy', save_best_only=True, verbose=1)
early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=1)

# Step 7: Train
model.fit(
    train_generator,
    validation_data=test_generator,
    epochs=EPOCHS,
    class_weight=class_weights,
    callbacks=[checkpoint, early_stop]
)

print(f"✅ Model saved to: {model_path}")
