import ntpath
import os

import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Convolution2D, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import pandas as pd
from imgaug import augmenters as iaa
import random


DATASETS = [("./data/track1_3laps", 400), ("./data/track1_5laps", 500), ("./data/track1_20laps_smooth", 4000), ("./data/track2_shadow_part", 100)]
DATASET_COLUMNS = ["center", "left", "right", "steering", "throttle", "reverse", "speed"]


def load_steering_img(data: np.ndarray, dataset: tuple[str, int]):
    image_paths = []
    steering = []
    for i in range(len(data)):
        indexed_data = data.iloc[i]
        center, left, right = indexed_data[0:3]
        for side in [(center, 0), (left, 0.15), (right, -0.15)]:
            image_paths.append(os.path.join(dataset[0], "IMG", side[0].strip()))
            steering.append(float(indexed_data[3]) + side[1])
    image_paths = np.asarray(image_paths)
    steering = np.asarray(steering)
    return image_paths, steering


def preprocess_data(dataset: tuple[str, int]):
    # Load data
    data = pd.read_csv(os.path.join(dataset[0], "driving_log.csv"), names=DATASET_COLUMNS)
    pd.set_option("max_columns", len(DATASET_COLUMNS))

    # Keep filename in paths
    for side in ["center", "left", "right"]:
        data[side] = data[side].apply(lambda p: ntpath.split(p)[1])

    # Display sample sizes for steering
    num_bins = 25
    samples_per_bin = dataset[1]

    # Display steering repartition
    hist, bins = np.histogram(data["steering"], num_bins)
    center = (bins[:-1] + bins[1:]) / 2
    plt.bar(center, hist, width=0.05)
    plt.plot((np.min(data["steering"]), np.max(data["steering"])), (samples_per_bin, samples_per_bin))
    plt.show()

    # Remove excess steering samples
    remove_list = []
    for j in range(num_bins):
        list_ = []
        for i in range(len(data["steering"])):
            if bins[j] <= data["steering"][i] <= bins[j + 1]:
                list_.append(i)
        remove_list.extend(shuffle(list_)[samples_per_bin:])
    data.drop(data.index[remove_list], inplace=True)

    hist, _ = np.histogram(data["steering"], num_bins)
    plt.bar(center, hist, width=0.05)
    plt.plot((np.min(data["steering"]), np.max(data["steering"])), (samples_per_bin, samples_per_bin))
    plt.show()

    # Load images
    image_paths, steering = load_steering_img(data, dataset)

    # Create training and validation sets
    x_train, x_valid, y_train, y_valid = train_test_split(image_paths, steering, test_size=0.2, random_state=6)
    print(f"Training samples : {len(x_train)}, validation samples : {len(x_valid)}")

    # Plotting sets
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].hist(y_train, bins=num_bins, width=0.05, color="blue")
    axes[0].set_title("Training set")
    axes[1].hist(y_valid, bins=num_bins, width=0.05, color="red")
    axes[1].set_title("Validation set")
    plt.show()

    image = image_paths[100]
    original_image = mpimg.imread(image)
    image = preprocess_img(image)

    # Plotting image example and the preprocessed image
    fig, axes = plt.subplots(1, 2, figsize=(15, 10))
    fig.tight_layout()
    axes[0].imshow(original_image)
    axes[0].set_title("Original image")
    axes[1].imshow(image)
    axes[1].set_title("Preprocessed image")
    plt.show()

    # Preprocessing all images
    x_train_gen, y_train_gen = next(batch_generator(x_train, y_train, 1, 1))
    x_valid_gen, y_valid_gen = next(batch_generator(x_valid, y_valid, 1, 0))

    # Display training image and validation image
    fig, axs = plt.subplots(1, 2, figsize=(15, 10))
    fig.tight_layout()
    axs[0].imshow(x_train_gen[0])
    axs[0].set_title("Training Image")
    axs[1].imshow(x_valid_gen[0])
    axs[1].set_title("Validation Image")
    plt.show()

    # Display data augmentation : zoom
    image = image_paths[random.randint(0, len(image_paths))]
    original_image = mpimg.imread(image)
    zoomed_image = zoom(original_image)
    fig, axs = plt.subplots(1, 2, figsize=(15, 10))
    fig.tight_layout()
    axs[0].imshow(original_image)
    axs[0].set_title("Original Image")
    axs[1].imshow(zoomed_image)
    axs[1].set_title("Zoomed Image")
    plt.show()

    # Display data augmentation : panning
    image = image_paths[random.randint(0, len(image_paths))]
    original_image = mpimg.imread(image)
    panned_image = pan(original_image)
    fig, axs = plt.subplots(1, 2, figsize=(15, 10))
    fig.tight_layout()
    axs[0].imshow(original_image)
    axs[0].set_title("Original Image")
    axs[1].imshow(panned_image)
    axs[1].set_title("Panned Image")
    plt.show()

    # Display data augmentation : brightness
    image = image_paths[random.randint(0, len(image_paths))]
    original_image = mpimg.imread(image)
    bright_image = img_random_brightness(original_image)
    fig, axs = plt.subplots(1, 2, figsize=(15, 10))
    fig.tight_layout()
    axs[0].imshow(original_image)
    axs[0].set_title("Original Image")
    axs[1].imshow(bright_image)
    axs[1].set_title("Bright Image")
    plt.show()

    # Display data augmentation : shadow
    image = image_paths[random.randint(0, len(image_paths))]
    original_image = mpimg.imread(image)
    shadow_image = img_random_shadow(original_image)
    fig, axs = plt.subplots(1, 2, figsize=(15, 10))
    fig.tight_layout()
    axs[0].imshow(original_image)
    axs[0].set_title("Original Image")
    axs[1].imshow(shadow_image)
    axs[1].set_title("Image with shadow")
    plt.show()

    # Display data augmentation : image flip
    random_index = random.randint(0, len(image_paths))
    image = image_paths[random_index]
    steering_angle = steering[random_index]
    original_image = mpimg.imread(image)
    flipped_image, flipped_angle = img_random_flip(original_image, steering_angle)
    fig, axs = plt.subplots(1, 2, figsize=(15, 10))
    fig.tight_layout()
    axs[0].imshow(original_image)
    axs[0].set_title("Original Image - " + "Steering Angle: " + str(steering_angle))
    axs[1].imshow(flipped_image)
    axs[1].set_title("Flipped Image" + "Steering Angle: " + str(flipped_angle))
    plt.show()

    # Display data augmentation : 10 random samples
    ncols = 2
    nrows = 10
    fig, axs = plt.subplots(nrows, ncols, figsize=(15, 50))
    fig.tight_layout()
    for i in range(10):
        rand_num = random.randint(0, len(image_paths) - 1)
        random_image = image_paths[rand_num]
        random_steering = steering[rand_num]
        original_image = mpimg.imread(random_image)
        augmented_image, steering_angle = random_augment(random_image, random_steering)
        axs[i][0].imshow(original_image)
        axs[i][0].set_title("Original Image")
        axs[i][1].imshow(augmented_image)
        axs[i][1].set_title("Augmented Image")
    plt.show()

    return x_train, y_train, x_valid, y_valid


def preprocess_img(img, is_np=False):
    if not is_np:
        img = mpimg.imread(img)
    # Crop image
    img = img[60:135, :, :]
    # Convert color to hsv
    img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    # Apply gaussian blur
    img = cv2.GaussianBlur(img, (3, 3), 0)
    # Resize image
    img = cv2.resize(img, (200, 66))
    # Normalize image
    img = img / 255
    return img


def nvidia_model() -> Sequential:
    """
    See https://arxiv.org/pdf/1604.07316v1.pdf
    """
    model = Sequential()
    model.add(Convolution2D(24, (5, 5), strides=(2, 2), input_shape=(66, 200, 3), activation="elu"))
    model.add(Convolution2D(36, (5, 5), strides=(2, 2), activation="elu"))
    model.add(Convolution2D(48, (5, 5), strides=(2, 2), activation="elu"))
    model.add(Convolution2D(64, (3, 3), activation="elu"))
    model.add(Convolution2D(64, (3, 3), activation="elu"))
    # model.add(Dropout(0.5))
    model.add(Flatten())
    model.add(Dropout(0.5))
    model.add(Dense(100, activation="elu"))
    # model.add(Dropout(0.5))
    model.add(Dense(50, activation="elu"))
    # model.add(Dropout(0.5))
    model.add(Dense(10, activation="elu"))
    # model.add(Dropout(0.5))
    model.add(Dense(1))
    optimizer = Adam(learning_rate=0.0001)
    model.compile(loss="mse", optimizer=optimizer, metrics=["accuracy"])
    return model


def zoom(image_to_zoom):
    zoom_func = iaa.Affine(scale=(1, 1.3))
    z_image = zoom_func.augment_image(image_to_zoom)
    return z_image


def pan(image_to_pan):
    pan_func = iaa.Affine(translate_percent={"x": (-0.1, 0.1), "y": (-0.1, 0.1)})
    pan_image = pan_func.augment_image(image_to_pan)
    return pan_image


def img_random_brightness(image_to_brighten):
    bright_func = iaa.Multiply((0.2, 1.2))
    bright_image = bright_func.augment_image(image_to_brighten)
    return bright_image


def img_random_shadow(image):
    image_with_shadow = image.copy()
    a = 1 if np.random.rand() >= 0.5 else -1
    b = round(np.random.rand() * 40) - 20
    for x in range(image_with_shadow.shape[0]):
        for y in range(image_with_shadow.shape[1]):
            if x * 2 * a + 40 + b < y:
                image_with_shadow[x, y] = image_with_shadow[x, y] * 0.4
    return image_with_shadow


def img_random_flip(image_to_flip, steering_angle):
    """Flips the image vertically"""
    flipped_image = cv2.flip(image_to_flip, 1)
    steering_angle = -steering_angle
    return flipped_image, steering_angle


def random_augment(image_to_augment, steering_angle):
    augment_image = mpimg.imread(image_to_augment)
    if np.random.rand() < 0.5:
        augment_image = zoom(augment_image)
    if np.random.rand() < 0.5:
        augment_image = pan(augment_image)
    if np.random.rand() < 0.5:
        augment_image = img_random_brightness(augment_image)
    if np.random.rand() < 0.5:
        augment_image = img_random_shadow(augment_image)
    if np.random.rand() < 0.5:
        augment_image, steering_angle = img_random_flip(augment_image, steering_angle)
    return augment_image, steering_angle


def batch_generator(image_paths, steering_ang, batch_size, is_training):
    while True:
        batch_img = []
        batch_steering = []
        for i in range(batch_size):
            random_index = random.randint(0, len(image_paths)-1)
            if is_training:
                im, steering = random_augment(image_paths[random_index], steering_ang[random_index])
            else:
                im = mpimg.imread(image_paths[random_index])
                steering = steering_ang[random_index]

            im = preprocess_img(im, True)
            batch_img.append(im)
            batch_steering.append(steering)
        yield np.asarray(batch_img), np.asarray(batch_steering)


def fit_model(model: Sequential, x_train, y_train, x_valid, y_valid):
    h = model.fit(batch_generator(x_train, y_train, 100, 1), steps_per_epoch=100,
                  epochs=20,
                  validation_data=batch_generator(x_valid, y_valid, 100, 0),
                  validation_steps=200,
                  verbose=1,
                  shuffle=1)
    plt.plot(h.history["loss"])
    plt.plot(h.history["val_loss"])
    plt.plot(h.history["accuracy"])
    plt.legend(["training", "validation", "accuracy"])
    plt.title("Loss")
    plt.xlabel("Epoch")
    plt.show()


def main():
    print("Building model...")
    model = nvidia_model()
    print(model.summary())

    for dataset in DATASETS:
        print(f"Preprocessing dataset \"{dataset[0]}\"...")
        x_train, y_train, x_valid, y_valid = preprocess_data(dataset)

        print("Fitting model...")
        fit_model(model, x_train, y_train, x_valid, y_valid)

    print("Saving model...")
    model.save("./out/model_hsv_large_track2.h5")


if __name__ == "__main__":
    main()
