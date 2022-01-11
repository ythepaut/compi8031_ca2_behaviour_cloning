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


DATASET_PATH = "./data/track2"
DATASET_COLUMNS = ["center", "left", "right", "steering", "throttle", "reverse", "speed"]


def load_steering_img(data):
    image_paths = []
    steering = []
    for i in range(len(data)):
        indexed_data = data.iloc[i]
        center, left, right = indexed_data[0:3]
        for side in [(center, 0), (left, 0.15), (right, -0.15)]:
            image_paths.append(os.path.join(DATASET_PATH, "IMG", side[0].strip()))
            steering.append(float(indexed_data[3]) + side[1])
    image_paths = np.asarray(image_paths)
    steering = np.asarray(steering)
    return image_paths, steering


def preprocess_data():
    # Load data
    data = pd.read_csv(os.path.join(DATASET_PATH, "driving_log.csv"), names=DATASET_COLUMNS)
    pd.set_option("max_columns", len(DATASET_COLUMNS))

    # Keep filename in paths
    for side in ["center", "left", "right"]:
        data[side] = data[side].apply(lambda p: ntpath.split(p)[1])

    # Display sample sizes for steering
    num_bins = 25
    samples_per_bin = 200

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
    image_paths, steering = load_steering_img(data)

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
    x_train = np.array(list(map(preprocess_img, x_train)))
    x_valid = np.array(list(map(preprocess_img, x_valid)))

    return x_train, y_train, x_valid, y_valid


def preprocess_img(img, is_np=False):
    if not is_np:
        img = mpimg.imread(img)
    # Crop image
    img = img[60:135, :, :]
    # Convert color to yuv
    img = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
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
    model.add(Convolution2D(36, (5, 5), strides=(2, 2), activation="relu"))
    model.add(Convolution2D(48, (5, 5), strides=(2, 2), activation="relu"))
    model.add(Convolution2D(64, (3, 3), activation="elu"))
    model.add(Convolution2D(64, (3, 3), activation="elu"))
    model.add(Dropout(0.5))
    model.add(Flatten())
    model.add(Dropout(0.5))
    model.add(Dense(100, activation="elu"))
    model.add(Dropout(0.5))
    model.add(Dense(50, activation="elu"))
    model.add(Dropout(0.5))
    model.add(Dense(10, activation="elu"))
    model.add(Dropout(0.5))
    model.add(Dense(1))
    optimizer = Adam(learning_rate=0.001)
    model.compile(loss="mse", optimizer=optimizer)
    return model


def fit_model(model: Sequential, x_train, y_train, x_valid, y_valid):
    h = model.fit(x_train, y_train, epochs=50, validation_data=(x_valid, y_valid), batch_size=100, verbose=1, shuffle=1)
    plt.plot(h.history["loss"])
    plt.plot(h.history["val_loss"])
    plt.title("Loss")
    plt.xlabel("Epoch")
    plt.show()


def main():
    print("Preprocessing...")
    x_train, y_train, x_valid, y_valid = preprocess_data()

    print("Building model...")
    model = nvidia_model()
    print(model.summary())

    print("Fitting model...")
    fit_model(model, x_train, y_train, x_valid, y_valid)

    print("Saving model...")
    model.save("./out/model_track2.h5")


if __name__ == "__main__":
    main()
