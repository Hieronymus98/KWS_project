2021.02.09 TUE
import json
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import timeit
from sklearn.model_selection import train_test_split

DATA_PATH = "data.json"
SAVED_MODEL_PATH = "DNN_model.h5"
EPOCHS = 40  # paper->20K
BATCH_SIZE = 100
PATIENCE = 5
LEARNING_RATE = 5*0.0001

def load_data(data_path):
    with open(data_path, "r") as fp:
        data=json.load(fp)
    
    x = np.array(data["MFCCs"])
    y = np.array(data["labels"])
    
    return x, y

def prepare_dataset(data_path, train_ratio = 0.8, test_ratio=0.1):
    validation_ratio = 1 - train_ratio - test_ratio
    
    x, y = load_data(data_path)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = test_ratio)
    x_train, x_validation, y_train, y_validation = train_test_split(x_train, y_train, test_size = validation_ratio/(train_ratio+validation_ratio))
    
    return x_train, y_train, x_validation, y_validation, x_test, y_test

def build_model(input_shape, learning_rate, loss = "sparse_categorical_crossentropy"):
    model = tf.keras.models.Sequential()
    
    model.add(tf.keras.layers.Flatten(input_shape=input_shape))
    model.add(tf.keras.layers.Dense(256, activation='relu'))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Dense(256, activation='relu'))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Dense(256, activation='relu'))
    model.add(tf.keras.layers.BatchNormalization())
    
    model.add(tf.keras.layers.Dense(10, activation='softmax'))
    optimiser = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    
    model.compile(optimizer=optimiser,
                  loss=loss,
                  metrics=["accuracy"])
    model.summary()
    return model
    
def train(model, epochs, batch_size, patience, x_train, y_train, x_validation, y_validation):

    earlystop_callback = tf.keras.callbacks.EarlyStopping(monitor="accuracy", min_delta=0.001, patience=patience)

    # train model
    history = model.fit(x_train,
                        y_train,
                        epochs=epochs,
                        batch_size=batch_size,
                        validation_data=(x_validation, y_validation),
                        callbacks=[earlystop_callback])
    return history

def plot_history(history):

    fig, axs = plt.subplots(2)

    # create accuracy subplot
    axs[0].plot(history.history["accuracy"], label="accuracy")
    axs[0].plot(history.history['val_accuracy'], label="val_accuracy")
    axs[0].set_ylabel("Accuracy")
    axs[0].legend(loc="lower right")
    axs[0].set_title("Accuracy evaluation")

    # create loss subplot
    axs[1].plot(history.history["loss"], label="loss")
    axs[1].plot(history.history['val_loss'], label="val_loss")
    axs[1].set_xlabel("Epoch")
    axs[1].set_ylabel("Loss")
    axs[1].legend(loc="upper right")
    axs[1].set_title("Loss evaluation")

    plt.show()

def main():
    x_train, y_train, x_validation, y_validation, x_test, y_test = prepare_dataset(DATA_PATH)
    input_shape = (x_train.shape[1], x_train.shape[2])
    model = build_model(input_shape, LEARNING_RATE)
    
    history = train(model, EPOCHS, BATCH_SIZE, PATIENCE, x_train, y_train, x_validation, y_validation)
    
    plot_history(history)
    
      # evaluate network on test set
    start_time = timeit.default_timer()
    test_loss, test_acc = model.evaluate(x_test, y_test)
    terminate_time = timeit.default_timer()
    
    latency = (terminate_time - start_time) / x_test.shape[0] * 1000
    print("\nTest loss: {}, test accuracy: {}".format(test_loss, 100*test_acc))
    print("\nlatency time: {}ms".format(latency))
    # save model
    model.save(SAVED_MODEL_PATH)

if __name__ == "__main__":
    main()
