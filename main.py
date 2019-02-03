import math
import numpy as np
import os
from keras.datasets import mnist
from keras.layers import Activation, BatchNormalization, Conv2D, Dense, Dropout, Flatten, LeakyReLU, ReLU, Reshape, UpSampling2D
from keras.models import load_model, Sequential
from keras.optimizers import Adam
from PIL import Image


def save_generated_images(generated_images, iteration):
    images_folder = "generated_images"
    if not os.path.exists(images_folder):
        os.mkdir(images_folder)
    batch_size = generated_images.shape[0]
    cols = int(math.sqrt(batch_size))
    rows = math.ceil(float(batch_size) / cols)
    width = generated_images.shape[2]
    height = generated_images.shape[1]
    image_blob = np.zeros((height * rows, width * cols), dtype=generated_images.dtype)
    for index, image in enumerate(generated_images):
        i = int(index / cols)
        j = index % cols
        image_blob[ width*i:width*(i+1), height*j:height*(j+1)] = image.reshape(28, 28)
    image_blob = (image_blob*127.5 + 127.5).astype("uint8")
    Image.fromarray(image_blob).save(os.path.join(images_folder, "generated_image_"+str(iteration)+".png"))



(x_train, _), (_, _) = mnist.load_data()
img_height = x_train.shape[1]
img_width = x_train.shape[2]

x_train = (x_train.reshape(-1, img_height, img_width, 1).astype("f") - 127.5) / 127.5

latent_dim = 100
generator = Sequential()
generator.add(Dense(1024, input_dim=latent_dim))
generator.add(BatchNormalization())
generator.add(ReLU())
generator.add(Dense(7*7*128))
generator.add(BatchNormalization())
generator.add(ReLU())
generator.add(Reshape((7, 7, 128)))
generator.add(UpSampling2D(2))
generator.add(Conv2D(64, 5, padding="same"))
generator.add(BatchNormalization())
generator.add(ReLU())
generator.add(UpSampling2D(2))
generator.add(Conv2D(1, 5, padding="same"))
generator.add(Activation("tanh"))


input_shape = (img_height, img_width, 1)
discriminator = Sequential()
discriminator.add(Conv2D(64, 5, strides=2, padding="same", input_shape=input_shape))
discriminator.add(LeakyReLU(0.2))
discriminator.add(Conv2D(128, 5, strides=2))
discriminator.add(LeakyReLU(0.2))
discriminator.add(Flatten())
discriminator.add(Dense(256))
discriminator.add(LeakyReLU(0.2))
discriminator.add(Dropout(0.5))
discriminator.add(Dense(1))
discriminator.add(Activation("sigmoid"))
discriminator.compile(Adam(lr=1e-5, beta_1=0.1), "binary_crossentropy")


discriminator.trainable = False
dcgan = Sequential([generator, discriminator])
dcgan.compile(Adam(lr=2e-4, beta_1=0.5), "binary_crossentropy")


generated_label = np.ones
real_label = np.zeros
start = 0
batch_size = 100
n_iterations = 10000
for iteration in range(1, n_iterations+1):
    random_latent_vectors = np.random.uniform(-1, 1, (batch_size, latent_dim))
    generated_images = generator.predict(random_latent_vectors)
    stop = start + batch_size
    real_images = x_train[start : stop]
    combined_images = np.concatenate([generated_images, real_images])
    labels = np.concatenate([generated_label((batch_size, 1)), real_label((batch_size, 1))])
    #labels += 0.05 * np.random.random(labels.shape) # Add random noise to the labels
    d_loss = discriminator.train_on_batch(combined_images, labels)
    random_latent_vectors = np.random.uniform(-1, 1, (batch_size, latent_dim))
    misleading_targets = real_label((batch_size, 1))
    g_loss = dcgan.train_on_batch(random_latent_vectors, misleading_targets)
    start += batch_size
    if start > len(x_train) - batch_size:
        start = 0
    if iteration % 1000 == 0:
        save_generated_images(generated_images, iteration)
    print("{} iteration elapsed  d_loss: {}  g_loss: {}".format(iteration, d_loss, g_loss))
    

models_folder = "models"
generator_path = os.path.join(models_folder, "generator.h5")
discriminator_path = os.path.join(models_folder, "discriminator.h5")
generator.save(generator_path)
discriminator.save(discriminator_path)