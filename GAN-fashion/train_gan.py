import os
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
from matplotlib import pyplot as plt

from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import (
    Conv2D,
    Dense,
    Flatten,
    Reshape,
    LeakyReLU,
    Dropout,
    UpSampling2D,
    Input,
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.preprocessing.image import array_to_img

# =========================
# Config
# =========================
LATENT_DIM = 128
BATCH_SIZE = 128
EPOCHS = 2000
IMAGE_DIR = "images"
MODEL_DIR = "models"

os.makedirs(IMAGE_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

# =========================
# GPU memory growth
# =========================
gpus = tf.config.experimental.list_physical_devices("GPU")
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

print("GPUs detected:", gpus)

# =========================
# Load dataset
# =========================
def scale_images(image, label):
    image = tf.cast(image, tf.float32) / 255.0
    image = tf.expand_dims(image, axis=-1) if len(image.shape) == 2 else image
    return image

ds, info = tfds.load(
    "fashion_mnist",
    split="train",
    as_supervised=True,
    with_info=True
)

print(info)

sample_image, sample_label = next(ds.as_numpy_iterator())
print("Sample label:", sample_label)
print("Sample image shape:", sample_image.shape)

ds = tfds.load("fashion_mnist", split="train", as_supervised=True)
ds = ds.map(scale_images, num_parallel_calls=tf.data.AUTOTUNE)
ds = ds.cache()
ds = ds.shuffle(60000)
ds = ds.batch(BATCH_SIZE)
ds = ds.prefetch(tf.data.AUTOTUNE)

# =========================
# Generator
# =========================
def build_gen():
    model = Sequential(name="generator")
    model.add(Input(shape=(LATENT_DIM,)))
    model.add(Dense(7 * 7 * 128))
    model.add(LeakyReLU(negative_slope=0.2))
    model.add(Reshape((7, 7, 128)))

    model.add(UpSampling2D())
    model.add(Conv2D(128, 5, padding="same"))
    model.add(LeakyReLU(negative_slope=0.2))

    model.add(UpSampling2D())
    model.add(Conv2D(128, 5, padding="same"))
    model.add(LeakyReLU(negative_slope=0.2))

    model.add(Conv2D(128, 4, padding="same"))
    model.add(LeakyReLU(negative_slope=0.2))

    model.add(Conv2D(128, 4, padding="same"))
    model.add(LeakyReLU(negative_slope=0.2))

    model.add(Conv2D(1, 4, padding="same", activation="sigmoid"))
    return model

# =========================
# Discriminator
# =========================
def build_disc():
    model = Sequential(name="discriminator")
    model.add(Input(shape=(28, 28, 1)))

    model.add(Conv2D(32, 5))
    model.add(LeakyReLU(negative_slope=0.2))
    model.add(Dropout(0.4))

    model.add(Conv2D(64, 5))
    model.add(LeakyReLU(negative_slope=0.2))
    model.add(Dropout(0.4))

    model.add(Conv2D(128, 5))
    model.add(LeakyReLU(negative_slope=0.2))
    model.add(Dropout(0.4))

    model.add(Conv2D(256, 5))
    model.add(LeakyReLU(negative_slope=0.2))
    model.add(Dropout(0.4))

    model.add(Flatten())
    model.add(Dropout(0.4))
    model.add(Dense(1, activation="sigmoid"))
    return model

generator = build_gen()
discriminator = build_disc()

generator.summary()
discriminator.summary()

# Quick test
test_noise = np.random.randn(4, LATENT_DIM).astype(np.float32)
generated = generator.predict(test_noise, verbose=0)
print("Generated shape:", generated.shape)

# =========================
# Losses and optimizers
# =========================
g_opt = Adam(learning_rate=0.0001)
d_opt = Adam(learning_rate=0.00001)
g_loss_fn = BinaryCrossentropy()
d_loss_fn = BinaryCrossentropy()

# =========================
# GAN model
# =========================
class FashionGAN(Model):
    def __init__(self, generator, discriminator, latent_dim=128, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.generator = generator
        self.discriminator = discriminator
        self.latent_dim = latent_dim

    def compile(self, g_opt, d_opt, g_loss, d_loss, *args, **kwargs):
        super().compile(*args, **kwargs)
        self.g_opt = g_opt
        self.d_opt = d_opt
        self.g_loss = g_loss
        self.d_loss = d_loss

    def train_step(self, batch):
        real_images = batch
        batch_size = tf.shape(real_images)[0]

        # -------------------------
        # Train discriminator
        # -------------------------
        random_latent_vectors = tf.random.normal((batch_size, self.latent_dim))
        fake_images = self.generator(random_latent_vectors, training=False)

        with tf.GradientTape() as d_tape:
            y_real = self.discriminator(real_images, training=True)
            y_fake = self.discriminator(fake_images, training=True)

            yhat_realfake = tf.concat([y_real, y_fake], axis=0)

            y_realfake = tf.concat(
                [tf.zeros_like(y_real), tf.ones_like(y_fake)], axis=0
            )

            noise_real = 0.15 * tf.random.uniform(tf.shape(y_real))
            noise_fake = -0.15 * tf.random.uniform(tf.shape(y_fake))
            y_realfake += tf.concat([noise_real, noise_fake], axis=0)

            total_d_loss = self.d_loss(y_realfake, yhat_realfake)

        dgrad = d_tape.gradient(total_d_loss, self.discriminator.trainable_variables)
        self.d_opt.apply_gradients(zip(dgrad, self.discriminator.trainable_variables))

        # -------------------------
        # Train generator
        # -------------------------
        random_latent_vectors = tf.random.normal((batch_size, self.latent_dim))

        with tf.GradientTape() as g_tape:
            gen_images = self.generator(random_latent_vectors, training=True)
            predicted_labels = self.discriminator(gen_images, training=False)
            total_g_loss = self.g_loss(tf.zeros_like(predicted_labels), predicted_labels)

        ggrad = g_tape.gradient(total_g_loss, self.generator.trainable_variables)
        self.g_opt.apply_gradients(zip(ggrad, self.generator.trainable_variables))

        return {"d_loss": total_d_loss, "g_loss": total_g_loss}

# =========================
# Callback
# =========================
class ModelMonitor(Callback):
    def __init__(self, num_img=3, latent_dim=128, save_interval=100):
        super().__init__()
        self.num_img = num_img
        self.latent_dim = latent_dim
        self.save_interval = save_interval

    def on_epoch_end(self, epoch, logs=None):
        if (epoch + 1) % self.save_interval != 0:
            return

        random_latent_vectors = tf.random.normal((self.num_img, self.latent_dim))
        generated_images = self.model.generator(random_latent_vectors)
        generated_images *= 255

        for i in range(self.num_img):
            img = array_to_img(generated_images[i])
            img.save(f"images/generated_epoch_{epoch+1}_{i}.png")

# =========================
# Train
# =========================
fash_gan = FashionGAN(generator, discriminator, latent_dim=LATENT_DIM)
fash_gan.compile(g_opt, d_opt, g_loss_fn, d_loss_fn)

hist = fash_gan.fit(
    ds,
    epochs=2000,
    callbacks=[ModelMonitor(num_img=4, save_interval=100)]
)

# =========================
# Save models
# =========================
generator.save(os.path.join(MODEL_DIR, "generator.h5"))
discriminator.save(os.path.join(MODEL_DIR, "discriminator.h5"))

# =========================
# Plot losses
# =========================
plt.title("Loss")
plt.plot(hist.history["d_loss"], label="d_loss")
plt.plot(hist.history["g_loss"], label="g_loss")
plt.legend()
plt.savefig(os.path.join(MODEL_DIR, "loss_plot.png"))
plt.show()

# =========================
# Generate sample grid
# =========================
sample_noise = tf.random.normal((16, LATENT_DIM))
imgs = generator.predict(sample_noise, verbose=0)

fig, ax = plt.subplots(ncols=4, nrows=4, figsize=(10, 10))
idx = 0
for r in range(4):
    for c in range(4):
        ax[r][c].imshow(np.squeeze(imgs[idx]), cmap="gray")
        ax[r][c].axis("off")
        idx += 1

plt.tight_layout()
plt.savefig(os.path.join(MODEL_DIR, "generated_grid.png"))
plt.show()
