# Import necessary libraries
import ix
import tensorflow as tf
import os
import random
import numpy as np
from tqdm import tqdm
from sklearn.metrics import precision_score, recall_score, f1_score
from skimage.io import imread, imshow
from skimage.transform import resize
import matplotlib.pyplot as plt

# Set a seed for reproducibility
seed = 42
np.random.seed = seed

# Define image dimensions and channels
IMG_WIDTH = 128
IMG_HEIGHT = 128
IMG_CHANNELS = 3

# Define paths for training and testing data
TRAIN_PATH = r"C:\Users\perfe\OneDrive - University of Greenwich\Desktop\U-net-checkKar\stage1_trains"
TEST_PATH = r"C:\Users\perfe\OneDrive - University of Greenwich\Desktop\U-net-checkKar\stage1_test"

# Check and print training directories
train_ids = next(os.walk(TRAIN_PATH), (None, None, []))[1]
if not train_ids:
    print("No training directories found in the specified path.")
else:
    print("Training directories found:", train_ids)

# Check and print test directories
test_ids = next(os.walk(TEST_PATH), (None, None, []))[1]
if not test_ids:
    print("No test directories found in the specified path.")
else:
    print("Test directories found:", test_ids)

# Initialize arrays to store training and test data
X_train = np.zeros((len(train_ids), IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), dtype=np.uint8)
Y_train = np.zeros((len(train_ids), IMG_HEIGHT, IMG_WIDTH, 1), dtype=bool)

# Load and preprocess training data
# from tensorflow.keras.preprocessing.image import ImageDataGenerator


for n, id_ in tqdm(enumerate(train_ids), total=len(train_ids)):
    images_dir = os.path.join(TRAIN_PATH, id_, 'images')
    image_files = [f for f in os.listdir(images_dir) if f.endswith('.png')]
    image_file = image_files[0]
    path = os.path.join(images_dir, image_file)
    img = imread(path)[:, :, :IMG_CHANNELS]
    img = resize(img, (IMG_HEIGHT, IMG_WIDTH), mode='constant', preserve_range=True)


    X_train[n] = img

    mask_dir = os.path.join(TRAIN_PATH, id_, 'masks')
    if os.path.exists(mask_dir) and os.path.isdir(mask_dir):
        mask = np.zeros((IMG_HEIGHT, IMG_WIDTH, 1), dtype=bool)
        mask_files = next(os.walk(mask_dir))[2]
        if mask_files:
            for mask_file in mask_files:
                mask_ = imread(os.path.join(mask_dir, mask_file))
                mask_ = np.expand_dims(resize(mask_, (IMG_HEIGHT, IMG_WIDTH), mode='constant', preserve_range=True), axis=-1)
                mask = np.maximum(mask, mask_)
            Y_train[n] = mask
        else:
            print(f"No mask files found for {id_}")
    else:
        print(f"Mask directory not found for {id_}")

# Load and preprocess test data
X_test = np.zeros((len(test_ids), IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), dtype=np.uint8)
sizes_test = []
print('Resizing test images')
for n, id_ in tqdm(enumerate(test_ids), total=len(test_ids)):
    path = os.path.join(TEST_PATH, id_)
    img_path = os.path.join(path, 'images', id_ + '.png')
    img = imread(img_path)[:, :, :IMG_CHANNELS]
    sizes_test.append([img.shape[0], img.shape[1]])
    img = resize(img, (IMG_HEIGHT, IMG_WIDTH), mode='constant', preserve_range=True)
    X_test[n] = img

print('Done!')

# Display random training and mask samples
image_x = random.randint(0, len(train_ids))
imshow(X_train[image_x])
plt.show()
imshow(np.squeeze(Y_train[image_x]))
plt.show()

# Build the U-net model
inputs = tf.keras.layers.Input((IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS))
s = tf.keras.layers.Lambda(lambda x: x / 255)(inputs)

# Contraction path
conv1 = tf.keras.layers.Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(s)
conv1 = tf.keras.layers.Dropout(0.1)(conv1)
conv1 = tf.keras.layers.Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(conv1)
p1 = tf.keras.layers.MaxPooling2D((2, 2))(conv1)

conv2 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p1)
conv2 = tf.keras.layers.Dropout(0.1)(conv2)
conv2 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(conv2)
p2 = tf.keras.layers.MaxPooling2D((2, 2))(conv2)

convolution3 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p2)
convolution3 = tf.keras.layers.Dropout(0.2)(convolution3)
convolution3 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(convolution3)
p3 = tf.keras.layers.MaxPooling2D((2, 2))(convolution3)

convolution4 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p3)
convolution4 = tf.keras.layers.Dropout(0.2)(convolution4)
convolution4 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(convolution4)
p4 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(convolution4)

convolution5 = tf.keras.layers.Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p4)
convolution5 = tf.keras.layers.Dropout(0.3)(convolution5)
convolution5 = tf.keras.layers.Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(convolution5)

# Expansive path
upsampling_6 = tf.keras.layers.Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(convolution5)
upsampling_6 = tf.keras.layers.concatenate([upsampling_6, convolution4])
U_convolution_6 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(upsampling_6)
U_convolution_6 = tf.keras.layers.Dropout(0.2)(U_convolution_6)
U_convolution_6 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(U_convolution_6)

upsampling_7 = tf.keras.layers.Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(U_convolution_6)
upsampling_7 = tf.keras.layers.concatenate([upsampling_7, convolution3])
U_convolution_7 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(upsampling_7)
U_convolution_7 = tf.keras.layers.Dropout(0.2)(U_convolution_7)
U_convolution_7 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(U_convolution_7)

upsampling_8 = tf.keras.layers.Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(U_convolution_7)
upsampling_8 = tf.keras.layers.concatenate([upsampling_8, conv2])
U_convolution_8 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(upsampling_8)
U_convolution_8 = tf.keras.layers.Dropout(0.1)(U_convolution_8)
U_convolution_8 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(U_convolution_8)

upsampling_9 = tf.keras.layers.Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same')(U_convolution_8)
upsampling_9 = tf.keras.layers.concatenate([upsampling_9, conv1], axis=3)
U_convolution_9 = tf.keras.layers.Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(upsampling_9)
U_convolution_9 = tf.keras.layers.Dropout(0.1)(U_convolution_9)
U_convolution_9 = tf.keras.layers.Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(U_convolution_9)

outputs = tf.keras.layers.Conv2D(1, (1, 1), activation='sigmoid')(U_convolution_9)

# Define and compile the model
model = tf.keras.Model(inputs=[inputs], outputs=[outputs])
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.summary()

# Model checkpoint and callbacks
checkpointer = tf.keras.callbacks.ModelCheckpoint('model_for_nuclei.h5', verbose=1, save_best_only=True)
callbacks = [
    tf.keras.callbacks.EarlyStopping(patience=2, monitor='val_loss'),
    tf.keras.callbacks.TensorBoard(log_dir='logs')]

# Train the model
results = model.fit(X_train, Y_train, validation_split=0.1, batch_size=16, epochs=25, callbacks=callbacks)

# Generate predictions on training, validation, and test sets
idx = random.randint(0, len(X_train))
preds_train = model.predict(X_train[:int(X_train.shape[0] * 0.9)], verbose=1)
preds_val = model.predict(X_train[int(X_train.shape[0] * 0.9):], verbose=1)
preds_test = model.predict(X_test, verbose=1)

# Threshold predictions to obtain binary masks
preds_train_t = (preds_train > 0.5).astype(np.uint8)
preds_val_t = (preds_val > 0.5).astype(np.uint8)
preds_test_t = (preds_test > 0.5).astype(np.uint8)

# Visualize random training and validation samples with predictions
ix_train = random.randint(0, len(preds_train_t))
ix_val = random.randint(0, len(preds_val_t))
imshow(X_train[ix_train])
plt.show()
imshow(np.squeeze(Y_train[ix_train]))
plt.show()
imshow(np.squeeze(preds_train_t[ix_train]))
plt.show()

imshow(X_train[int(X_train.shape[0] * 0.9):][ix_val])
plt.show()
imshow(np.squeeze(Y_train[int(Y_train.shape[0] * 0.9):][ix_val]))
plt.show()
imshow(np.squeeze(preds_val_t[ix_val]))
plt.show()


def calculate_iou(y_true, y_pred):
    intersection = np.sum(np.logical_and(y_true, y_pred))
    union = np.sum(np.logical_or(y_true, y_pred))

    iou = intersection / union
    return iou


# Assuming preds_train_t[ix] and Y_train[ix] are the predicted and true masks for a specific sample
iou = calculate_iou(np.squeeze(Y_train[ix]), np.squeeze(preds_train_t[ix]))
print(f'IoU for the sample: {iou}')

# Calculate IoU for random training sample
iou_train = calculate_iou(np.squeeze(Y_train[ix]), np.squeeze(preds_train_t[ix]))
print(f'IoU for random training sample: {iou_train}')

# Calculate IoU for random validation sample
ix_val = random.randint(0, len(preds_val_t))
iou_val = calculate_iou(np.squeeze(Y_train[int(X_train.shape[0] * 0.9):][ix_val]), np.squeeze(preds_val_t[ix_val]))
print(f'IoU for random validation sample: {iou_val}')


def calculate_metrics(y_true, y_pred):
    # Flatten the masks
    y_true_flat = y_true.flatten()
    y_pred_flat = y_pred.flatten()

    # Convert to binary values
    y_true_bin = (y_true_flat > 0.5).astype(int)
    y_pred_bin = (y_pred_flat > 0.5).astype(int)

    # Calculate metrics
    precision = precision_score(y_true_bin, y_pred_bin)
    recall = recall_score(y_true_bin, y_pred_bin)
    f1 = f1_score(y_true_bin, y_pred_bin)

    return precision, recall, f1

def dice_coefficient(y_true, y_pred):
    intersection = np.sum(np.logical_and(y_true, y_pred))
    union = np.sum(np.logical_or(y_true, y_pred))
    dice = 2 * intersection / (intersection + union)
    return dice

# Assuming preds_train_t[ix] and Y_train[ix] are the predicted and true masks for a specific sample
precision, recall, f1 = calculate_metrics(np.squeeze(Y_train[ix]), np.squeeze(preds_train_t[ix]))
dice = dice_coefficient(np.squeeze(Y_train[ix]), np.squeeze(preds_train_t[ix]))

print(f'Precision for the sample: {precision}')
print(f'Recall for the sample: {recall}')
print(f'F1 Score for the sample: {f1}')
print(f'Dice Coefficient for the sample: {dice}')

# Calculate metrics for random training sample
precision_train, recall_train, f1_train = calculate_metrics(np.squeeze(Y_train[ix]), np.squeeze(preds_train_t[ix]))
dice_train = dice_coefficient(np.squeeze(Y_train[ix]), np.squeeze(preds_train_t[ix]))
print(f'Precision for random training sample: {precision_train}')
print(f'Recall for random training sample: {recall_train}')
print(f'F1 Score for random training sample: {f1_train}')
print(f'Dice Coefficient for random training sample: {dice_train}')

# Calculate metrics for random validation sample
ix_val = random.randint(0, len(preds_val_t))
precision_val, recall_val, f1_val = calculate_metrics(np.squeeze(Y_train[int(X_train.shape[0] * 0.9):][ix_val]),
                                                      np.squeeze(preds_val_t[ix_val]))
dice_val = dice_coefficient(np.squeeze(Y_train[int(X_train.shape[0] * 0.9):][ix_val]), np.squeeze(preds_val_t[ix_val]))

print(f'Precision for random validation sample: {precision_val}')
print(f'Recall for random validation sample: {recall_val}')
print(f'F1 Score for random validation sample: {f1_val}')
print(f'Dice Coefficient for random validation sample: {dice_val}')