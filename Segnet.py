import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Conv2D, MaxPooling2D,
    Conv2DTranspose, UpSampling2D, concatenate
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau
from sklearn.model_selection import train_test_split

# --------------------
# CONFIG
# --------------------
DATA_DIR = "/Users/sabhyatasinha/HYBRID/split/test"
IMG_SIZE = 256
BATCH_SIZE = 8
EPOCHS = 30
LR = 3e-4
SEED = 42

tf.random.set_seed(SEED)
np.random.seed(SEED)

# --------------------
# METRICS & LOSSES
# --------------------
def pixel_accuracy(y_true, y_pred):
    y_pred = tf.round(y_pred)
    y_true = tf.round(y_true)
    correct = tf.equal(y_true, y_pred)
    return tf.reduce_mean(tf.cast(correct, tf.float32))

def dice_coef(y_true, y_pred, smooth=1e-6):
    y_true_f = tf.reshape(y_true, [-1])
    y_pred_f = tf.reshape(y_pred, [-1])
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (
        tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth
    )

def dice_loss(y_true, y_pred):
    return 1.0 - dice_coef(y_true, y_pred)

def bce_dice_loss(y_true, y_pred):
    bce = tf.keras.losses.binary_crossentropy(y_true, y_pred)
    return bce + dice_loss(y_true, y_pred)

def iou_coef(y_true, y_pred, smooth=1e-6):
    y_true_f = tf.reshape(y_true, [-1])
    y_pred_f = tf.reshape(y_pred, [-1])
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    union = tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) - intersection
    return (intersection + smooth) / (union + smooth)

# --------------------
# LOAD DATA
# --------------------
def load_data(data_dir, img_size):
    images_path = os.path.join(data_dir, "images")
    masks_path = os.path.join(data_dir, "masks")

    image_files = sorted(os.listdir(images_path))
    mask_files = sorted(os.listdir(masks_path))

    X, Y = [], []

    for img_file, mask_file in zip(image_files, mask_files):
        img = tf.keras.preprocessing.image.load_img(
            os.path.join(images_path, img_file),
            target_size=(img_size, img_size)
        )
        mask = tf.keras.preprocessing.image.load_img(
            os.path.join(masks_path, mask_file),
            target_size=(img_size, img_size),
            color_mode="grayscale"
        )

        X.append(np.array(img) / 255.0)
        Y.append(np.array(mask) / 255.0)

    X = np.array(X, dtype=np.float32)
    Y = np.array(Y, dtype=np.float32)
    Y = np.expand_dims(Y, axis=-1)

    return X, Y

X, Y = load_data(DATA_DIR, IMG_SIZE)

X_train, X_val, Y_train, Y_val = train_test_split(
    X, Y, test_size=0.1, random_state=SEED
)

print("Train:", X_train.shape, Y_train.shape)
print("Val:", X_val.shape, Y_val.shape)


# --------------------
# SEGNET MODEL
# --------------------
def segnet_model(input_size=(IMG_SIZE, IMG_SIZE, 3)):
    inputs = Input(input_size)

    # Encoder
    x = Conv2D(64, 3, activation='relu', padding='same')(inputs)
    x = Conv2D(64, 3, activation='relu', padding='same')(x)
    x = MaxPooling2D(2)(x)

    x = Conv2D(128, 3, activation='relu', padding='same')(x)
    x = Conv2D(128, 3, activation='relu', padding='same')(x)
    x = MaxPooling2D(2)(x)

    x = Conv2D(256, 3, activation='relu', padding='same')(x)
    x = Conv2D(256, 3, activation='relu', padding='same')(x)
    x = MaxPooling2D(2)(x)

    x = Conv2D(512, 3, activation='relu', padding='same')(x)
    x = Conv2D(512, 3, activation='relu', padding='same')(x)
    x = MaxPooling2D(2)(x)

    # Decoder
    x = UpSampling2D(2)(x)
    x = Conv2D(512, 3, activation='relu', padding='same')(x)
    x = Conv2D(512, 3, activation='relu', padding='same')(x)

    x = UpSampling2D(2)(x)
    x = Conv2D(256, 3, activation='relu', padding='same')(x)
    x = Conv2D(256, 3, activation='relu', padding='same')(x)

    x = UpSampling2D(2)(x)
    x = Conv2D(128, 3, activation='relu', padding='same')(x)
    x = Conv2D(128, 3, activation='relu', padding='same')(x)

    x = UpSampling2D(2)(x)
    x = Conv2D(64, 3, activation='relu', padding='same')(x)
    x = Conv2D(64, 3, activation='relu', padding='same')(x)

    outputs = Conv2D(1, 1, activation='sigmoid')(x)

    return Model(inputs, outputs)

# --------------------
# CHOOSE MODEL
# --------------------
# model = unet_model()
model = segnet_model()

model.summary()

# --------------------
# COMPILE
# --------------------
model.compile(
    optimizer=Adam(LR),
    loss=bce_dice_loss,
    metrics=[pixel_accuracy, dice_coef, iou_coef]
)

# --------------------
# DATA PIPELINE
# --------------------
def augment(img, mask):
    if tf.random.uniform(()) > 0.5:
        img = tf.image.flip_left_right(img)
        mask = tf.image.flip_left_right(mask)
    if tf.random.uniform(()) > 0.5:
        img = tf.image.flip_up_down(img)
        mask = tf.image.flip_up_down(mask)
    return img, mask

train_ds = tf.data.Dataset.from_tensor_slices((X_train, Y_train))
train_ds = train_ds.shuffle(256, seed=SEED)
train_ds = train_ds.map(augment, num_parallel_calls=tf.data.AUTOTUNE)
train_ds = train_ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

val_ds = tf.data.Dataset.from_tensor_slices((X_val, Y_val))
val_ds = val_ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

# --------------------
# CALLBACK
# --------------------
lr_callback = ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.5,
    patience=4,
    verbose=1
)

# --------------------
# TRAIN
# --------------------
history = model.fit(
    train_ds,
    epochs=EPOCHS,
    validation_data=val_ds,
    callbacks=[lr_callback],
    verbose=1
)

# --------------------
# SAVE
# --------------------
model.save("segnet_bce_dice.keras")
np.save("training_history.npy", history.history)
print("✅ Training complete & model saved")

# --------------------
# PLOTS
# --------------------
plt.figure(figsize=(18, 5))

plt.subplot(1, 3, 1)
plt.plot(history.history["loss"], label="Train")
plt.plot(history.history["val_loss"], label="Val")
plt.title("BCE + Dice Loss")
plt.legend()
plt.grid(alpha=0.4)

plt.subplot(1, 3, 2)
plt.plot(history.history["dice_coef"], label="Train")
plt.plot(history.history["val_dice_coef"], label="Val")
plt.title("Dice Coefficient")
plt.legend()
plt.grid(alpha=0.4)

plt.subplot(1, 3, 3)
plt.plot(history.history["iou_coef"], label="Train")
plt.plot(history.history["val_iou_coef"], label="Val")
plt.title("IoU")
plt.legend()
plt.grid(alpha=0.4)

plt.tight_layout()
plt.show()