import os
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras import layers, Model

# Check that a GPU is available.
gpu_devices = tf.config.list_physical_devices('GPU')
if not gpu_devices:
    raise EnvironmentError("No GPU available. This training requires a CUDA-enabled GPU.")
else:
    print("Num GPUs Available: ", len(gpu_devices))
    for gpu in gpu_devices:
        tf.config.experimental.set_memory_growth(gpu, True)

# -----------------------------
# Dataset Loader for Single-Folder Word Images
# -----------------------------
def load_word_image(image_path, img_size=(64, 256)):
    # Note: img_size is (height, width) but PIL expects (width, height), so we swap.
    image = Image.open(image_path).convert('L')
    image = image.resize((img_size[1], img_size[0]))  # (width, height)
    image = np.array(image).astype(np.float32)
    image = image / 127.5 - 1.  # Normalize to [-1, 1]
    image = np.expand_dims(image, axis=-1)  # Shape: (height, width, 1)
    return image

def load_word_dataset(root_dir, img_size=(64,256)):
    """
    Loads images from a single folder.
    Each image file is treated as a unique label.
    For example:
        words/banana.png -> label "banana"
    """
    images = []
    label_names = []
    
    file_list = sorted(
        f for f in os.listdir(root_dir)
        if f.lower().endswith(('.png', '.jpg', '.jpeg'))
    )
    
    for fname in file_list:
        path = os.path.join(root_dir, fname)
        img = load_word_image(path, img_size)
        images.append(img)
        label_base, _ = os.path.splitext(fname)
        label_names.append(label_base)
    
    unique_labels = sorted(set(label_names))
    label_to_int = {label: i for i, label in enumerate(unique_labels)}
    labels = [label_to_int[lbl] for lbl in label_names]
    
    images = np.array(images)
    labels = np.array(labels)
    return images, labels, len(unique_labels)

# -----------------------------
# Build the Generator for Word-level Images
# -----------------------------
def build_generator(z_dim, n_classes, img_shape):
    noise = layers.Input(shape=(z_dim,))
    label = layers.Input(shape=(1,), dtype='int32')
    # Embed the label and multiply with the noise vector
    label_embedding = layers.Embedding(n_classes, z_dim)(label)
    label_embedding = layers.Flatten()(label_embedding)
    model_input = layers.multiply([noise, label_embedding])
    # Project and reshape
    x = layers.Dense(256 * (img_shape[0]//4) * (img_shape[1]//4))(model_input)
    x = layers.Reshape((img_shape[0]//4, img_shape[1]//4, 256))(x)
    x = layers.UpSampling2D()(x)
    x = layers.Conv2D(256, kernel_size=3, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(negative_slope=0.2)(x)
    x = layers.UpSampling2D()(x)
    x = layers.Conv2D(128, kernel_size=3, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(negative_slope=0.2)(x)
    x = layers.Conv2D(img_shape[2], kernel_size=3, padding="same", activation="tanh")(x)
    model = Model([noise, label], x)
    return model

# -----------------------------
# Build the Discriminator for Word-level Images
# -----------------------------
def build_discriminator(n_classes, img_shape):
    img = layers.Input(shape=img_shape)
    label = layers.Input(shape=(1,), dtype='int32')
    label_embedding = layers.Embedding(n_classes, np.prod(img_shape))(label)
    label_embedding = layers.Flatten()(label_embedding)
    label_embedding = layers.Reshape(img_shape)(label_embedding)
    # Concatenate image and label embedding along the channel axis
    merged = layers.Concatenate()([img, label_embedding])
    x = layers.Conv2D(64, kernel_size=3, strides=2, padding="same")(merged)
    x = layers.LeakyReLU(negative_slope=0.2)(x)
    x = layers.Dropout(0.25)(x)
    x = layers.Conv2D(128, kernel_size=3, strides=2, padding="same")(x)
    x = layers.ZeroPadding2D(padding=((0,1),(0,1)))(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(negative_slope=0.2)(x)
    x = layers.Dropout(0.25)(x)
    x = layers.Conv2D(256, kernel_size=3, strides=2, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(negative_slope=0.2)(x)
    x = layers.Dropout(0.25)(x)
    x = layers.Flatten()(x)
    validity = layers.Dense(1, activation='sigmoid')(x)
    model = Model([img, label], validity)
    return model

# -----------------------------
# Training the Word-level cGAN (TensorFlow/Keras)
# -----------------------------
def train_cgan_tensorflow():
    # Hyperparameters
    epochs = 1000  # Adjust as needed
    batch_size = 4   # With few samples, a small batch size is used.
    z_dim = 100
    img_shape = (64, 256, 1)  # (height, width, channels)

    images, labels, n_classes = load_word_dataset("words", img_size=(img_shape[0], img_shape[1]))
    dataset_size = len(images)
    print(f"Loaded {dataset_size} images with {n_classes} unique labels.")

    dataset = tf.data.Dataset.from_tensor_slices((images, labels)).shuffle(dataset_size).batch(batch_size)
    steps_per_epoch = dataset_size // batch_size

    # Create separate optimizers for generator and discriminator.
    gen_optimizer = tf.keras.optimizers.Adam(0.0002, beta_1=0.5)
    disc_optimizer = tf.keras.optimizers.Adam(0.0002, beta_1=0.5)
    loss_fn = tf.keras.losses.BinaryCrossentropy()

    generator = build_generator(z_dim, n_classes, img_shape)
    discriminator = build_discriminator(n_classes, img_shape)

    # Dummy forward pass to build variables before training.
    dummy_noise = tf.zeros((1, z_dim))
    dummy_label = tf.zeros((1,), dtype=tf.int32)
    dummy_gen = generator([dummy_noise, dummy_label])
    _ = discriminator([dummy_gen, dummy_label])

    @tf.function
    def train_step(real_images, real_labels):
        batch_size_current = tf.shape(real_images)[0]
        valid = tf.ones((batch_size_current, 1))
        fake = tf.zeros((batch_size_current, 1))
        noise = tf.random.normal((batch_size_current, z_dim))

        # Train Generator
        with tf.GradientTape() as tape:
            gen_images = generator([noise, real_labels], training=True)
            validity = discriminator([gen_images, real_labels], training=True)
            g_loss = loss_fn(valid, validity)
        gen_grads = tape.gradient(g_loss, generator.trainable_variables)
        gen_optimizer.apply_gradients(zip(gen_grads, generator.trainable_variables))

        # Train Discriminator
        with tf.GradientTape() as tape:
            validity_real = discriminator([real_images, real_labels], training=True)
            d_real_loss = loss_fn(valid, validity_real)
            validity_fake = discriminator([gen_images, real_labels], training=True)
            d_fake_loss = loss_fn(fake, validity_fake)
            d_loss = (d_real_loss + d_fake_loss) / 2
        disc_grads = tape.gradient(d_loss, discriminator.trainable_variables)
        disc_optimizer.apply_gradients(zip(disc_grads, discriminator.trainable_variables))
        return g_loss, d_loss

    for epoch in range(epochs):
        iterator = iter(dataset)  # Reinitialize the dataset iterator each epoch.
        for step in range(steps_per_epoch):
            real_images, real_labels = next(iterator)
            g_loss, d_loss = train_step(real_images, real_labels)
        if epoch % 50 == 0:
            tf.print("Epoch:", epoch, "Generator Loss:", g_loss, "Discriminator Loss:", d_loss)
    generator.save("generator_cgan_tf_words.h5")
    print("Training complete. Generator saved to generator_cgan_tf_words.h5")

if __name__ == "__main__":
    # Run strictly on GPU.
    with tf.device('/GPU:0'):
        train_cgan_tensorflow()
