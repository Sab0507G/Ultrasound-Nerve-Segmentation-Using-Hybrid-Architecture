# ============================================================
# Hybrid U-Net Encoder + SegNet Decoder (NO Attention Gates)
# Focal Tversky Loss | Dice | Foreground IoU
# ============================================================

# ------------------- IMPORTS -------------------
import os, glob, random
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, callbacks
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# ------------------- CONFIG -------------------
DATA_DIR   = "/Users/sabhyatasinha/HYBRID/split/test"
IMG_SIZE   = 256
BATCH_SIZE = 8
EPOCHS     = 30
INITIAL_LR = 3e-4
SEED       = 14

os.environ["PYTHONHASHSEED"] = str(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

# ------------------- LOSSES & METRICS -------------------
@tf.keras.utils.register_keras_serializable()
def dice_coef(y_true, y_pred, smooth=1e-6):
    y_true_f = tf.reshape(y_true, [-1])
    y_pred_f = tf.reshape(y_pred, [-1])
    inter = tf.reduce_sum(y_true_f * y_pred_f)
    return (2. * inter + smooth) / (
        tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth
    )

@tf.keras.utils.register_keras_serializable()
def tversky_loss(y_true, y_pred, alpha=0.7, beta=0.3, smooth=1e-6):
    y_true_f = tf.reshape(y_true, [-1])
    y_pred_f = tf.reshape(y_pred, [-1])
    TP = tf.reduce_sum(y_true_f * y_pred_f)
    FN = tf.reduce_sum(y_true_f * (1 - y_pred_f))
    FP = tf.reduce_sum((1 - y_true_f) * y_pred_f)
    return 1 - (TP + smooth) / (TP + alpha * FN + beta * FP + smooth)

@tf.keras.utils.register_keras_serializable()
def focal_tversky_loss(y_true, y_pred):
    return tf.pow(tversky_loss(y_true, y_pred), 1.5)

class ForegroundIoU(tf.keras.metrics.Metric):
    def __init__(self, name="foreground_iou", threshold=0.5, **kwargs):
        super().__init__(name=name, **kwargs)
        self.threshold = threshold
        self.inter = self.add_weight(initializer="zeros")
        self.union = self.add_weight(initializer="zeros")

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.cast(y_true > 0.5, tf.float32)
        y_pred = tf.cast(y_pred > self.threshold, tf.float32)
        inter = tf.reduce_sum(y_true * y_pred)
        union = tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) - inter
        self.inter.assign_add(inter)
        self.union.assign_add(union)

    def result(self):
        return tf.math.divide_no_nan(self.inter, self.union)

    def reset_states(self):
        self.inter.assign(0.)
        self.union.assign(0.)

# ------------------- DATA LOADING -------------------
def get_paths(data_dir):
    img_dir  = os.path.join(data_dir, "images")
    mask_dir = os.path.join(data_dir, "masks")

    image_files = sorted([
        os.path.join(img_dir, f)
        for f in os.listdir(img_dir)
        if f.lower().endswith(('.png','.jpg','.jpeg','.bmp','.tif','.tiff'))
    ])

    mask_files = sorted([
        os.path.join(mask_dir, f)
        for f in os.listdir(mask_dir)
        if f.lower().endswith(('.png','.jpg','.jpeg','.bmp','.tif','.tiff'))
    ])

    if len(image_files) == 0 or len(mask_files) == 0:
        raise RuntimeError("No images or masks found")

    if len(image_files) != len(mask_files):
        raise RuntimeError("Image-mask count mismatch")

    return image_files, mask_files

image_paths, mask_paths = get_paths(DATA_DIR)

train_img, val_img, train_mask, val_mask = train_test_split(
    image_paths, mask_paths,
    test_size=0.1,
    random_state=SEED,
    shuffle=True
)

# ------------------- PREPROCESS -------------------
def normalize(img):
    img = img.astype(np.float32)
    return (img - img.mean()) / (img.std() + 1e-5)

def read_pair(img_p, mask_p):
    img = tf.image.resize(
        tf.image.decode_image(tf.io.read_file(img_p), 3),
        [IMG_SIZE, IMG_SIZE]
    )
    img = normalize(img.numpy())

    mask = tf.image.resize(
        tf.image.decode_image(tf.io.read_file(mask_p), 1),
        [IMG_SIZE, IMG_SIZE],
        method="nearest"
    )
    mask = (mask.numpy() / 255.0 > 0.5).astype(np.float32)

    return img, mask

def data_gen(imgs, masks, augment=False):
    aug = ImageDataGenerator(
        rotation_range=5,
        zoom_range=0.05,
        width_shift_range=0.03,
        height_shift_range=0.03
    )
    while True:
        idx = np.random.permutation(len(imgs))
        for i in idx:
            img, mask = read_pair(imgs[i], masks[i])
            if augment:
                seed = random.randint(0, 1_000_000)
                img  = aug.random_transform(img, seed=seed)
                mask = aug.random_transform(mask, seed=seed)
            yield img, mask

def make_ds(imgs, masks, augment=False):
    ds = tf.data.Dataset.from_generator(
        lambda: data_gen(imgs, masks, augment),
        output_signature=(
            tf.TensorSpec((IMG_SIZE, IMG_SIZE, 3), tf.float32),
            tf.TensorSpec((IMG_SIZE, IMG_SIZE, 1), tf.float32),
        )
    )
    return ds.batch(BATCH_SIZE).repeat().prefetch(tf.data.AUTOTUNE)

train_ds = make_ds(train_img, train_mask, augment=True)
val_ds   = make_ds(val_img, val_mask)

# ------------------- MODEL -------------------
def conv_block(x, f):
    x = layers.Conv2D(f, 3, padding="same", kernel_initializer="he_normal")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.Conv2D(f, 3, padding="same", kernel_initializer="he_normal")(x)
    x = layers.BatchNormalization()(x)
    return layers.Activation("relu")(x)

def build_model():
    inputs = layers.Input((IMG_SIZE, IMG_SIZE, 3))

    # Encoder (U-Net)
    c1 = conv_block(inputs, 64);   p1 = layers.MaxPooling2D()(c1)
    c2 = conv_block(p1, 128);      p2 = layers.MaxPooling2D()(c2)
    c3 = conv_block(p2, 256);      p3 = layers.MaxPooling2D()(c3)
    c4 = conv_block(p3, 512);      p4 = layers.MaxPooling2D()(c4)
    c5 = conv_block(p4, 1024)

    # Decoder (SegNet-style, NO attention)
    d4 = layers.UpSampling2D()(c5)
    d4 = conv_block(layers.Concatenate()([d4, c4]), 512)

    d3 = layers.UpSampling2D()(d4)
    d3 = conv_block(layers.Concatenate()([d3, c3]), 256)

    d2 = layers.UpSampling2D()(d3)
    d2 = conv_block(layers.Concatenate()([d2, c2]), 128)

    d1 = layers.UpSampling2D()(d2)
    d1 = conv_block(layers.Concatenate()([d1, c1]), 64)

    outputs = layers.Conv2D(1, 1, activation="sigmoid")(d1)
    return models.Model(inputs, outputs, name="UNet_SegNet_NoAttention")

model = build_model()

model.compile(
    optimizer=optimizers.Adam(INITIAL_LR, clipnorm=1.0),
    loss=focal_tversky_loss,
    metrics=[dice_coef, "accuracy", ForegroundIoU()]
)

model.summary()

# ------------------- TRAIN -------------------
steps = max(1, len(train_img) // BATCH_SIZE)
val_steps = max(1, len(val_img) // BATCH_SIZE)

callbacks_list = [
    callbacks.ModelCheckpoint(
        "best_model.keras",
        monitor="val_dice_coef",
        save_best_only=True,
        mode="max"
    ),
    callbacks.ReduceLROnPlateau(
        monitor="val_dice_coef",
        factor=0.5,
        patience=4,
        mode="max"
    ),
    callbacks.EarlyStopping(
        monitor="val_dice_coef",
        patience=8,
        restore_best_weights=True,
        mode="max"
    )
]

history = model.fit(
    train_ds,
    validation_data=val_ds,
    steps_per_epoch=steps,
    validation_steps=val_steps,
    epochs=EPOCHS,
    callbacks=callbacks_list,
    verbose=1
)

model.save("final_unet_segnet_no_attention.keras")
print("✅ TRAINING COMPLETE")

# ------------------- PLOTS -------------------
plt.figure(figsize=(18, 5))

plt.subplot(1, 3, 1)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title("Focal Tversky Loss")
plt.grid(True)

plt.subplot(1, 3, 2)
plt.plot(history.history['dice_coef'])
plt.plot(history.history['val_dice_coef'])
plt.title("Dice Coefficient")
plt.grid(True)

plt.subplot(1, 3, 3)
plt.plot(history.history['foreground_iou'])
plt.plot(history.history['val_foreground_iou'])
plt.title("Foreground IoU")
plt.grid(True)

plt.tight_layout()
plt.show()

np.save("training_history.npy", history.history)
print("✅ HISTORY SAVED")