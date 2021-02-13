# 2021.02.09 TUE
import json
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import timeit
from sklearn.model_selection import train_test_split

DATA_PATH = "data.json"
SAVED_MODEL_PATH = "CRNN_model.h5"
EPOCHS = 40  # paper->20K
BATCH_SIZE = 100
PATIENCE = 5
LEARNING_RATE = 5*0.0001  # become 0.0001 after 10K iterations in paper


def load_data(data_path):
    with open(data_path, "r") as fp:
        data = json.load(fp)
    
    X = np.array(data["MFCCs"])
    y = np.array(data["labels"])
    print("Training sets loaded!")
    # print(X.shape[0], X.shape[1], X.shape[2])
    # print(X)
    return X, y


def prepare_dataset(data_path, test_size=0.2, validation_size=0.2):

    # load dataset
    X, y = load_data(data_path)

    # create train, validation, test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)
    X_train, X_validation, y_train, y_validation = train_test_split(X_train, y_train, test_size=validation_size)

    # add an axis to nd array (convert 2D to 3D)
    X_train = X_train[..., np.newaxis]
    X_test = X_test[..., np.newaxis]
    X_validation = X_validation[..., np.newaxis]

    # print(X_train.shape[0], X_train.shape[1], X_train.shape[2], X_train.shape[3])
    # print(X_train)
    return X_train, y_train, X_validation, y_validation, X_test, y_test


def build_model(input_shape, loss="sparse_categorical_crossentropy", learning_rate=0.0001):

    # build network architecture using convolutional layers
    model = tf.keras.models.Sequential()

    # 1st conv layer
    # (# of filter, size of filter, )
    model.add(tf.keras.layers.Conv2D(128, kernel_size = (4, 4), strides = (2, 2), activation='relu', input_shape=input_shape, 
                                  kernel_regularizer=tf.keras.regularizers.l2(0.001)))
    model.add(tf.keras.layers.BatchNormalization())  
    model.add(tf.keras.layers.MaxPooling2D((3, 3), strides=(2,2), padding='same')) # (pool size, ...)

    model.summary()
    print(model.output_shape[0], model.output_shape[1], model.output_shape[2], model.output_shape[3])
    model.add(tf.keras.layers.Reshape((-1, 128)))  # transform data dimension to be compatible with next layer
    
    model.summary()
    # model.add(tf.keras.layers.GRU(units=76, batch_input_shape=input_size, return_sequences=True))
    model.add(tf.keras.layers.GRU(units=76, return_sequences=True))
    model.add(tf.keras.layers.GRU(units=76))
    
    model.add(tf.keras.layers.Dense(164, activation='relu'))

    # softmax output layer
    model.add(tf.keras.layers.Dense(10, activation='softmax'))

    optimiser = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    # compile model
    model.compile(optimizer=optimiser,
                  loss=loss,
                  metrics=["accuracy"])

    # print model parameters on console
    model.summary()

    return model


def train(model, epochs, batch_size, patience, X_train, y_train, X_validation, y_validation):

    earlystop_callback = tf.keras.callbacks.EarlyStopping(monitor="accuracy", min_delta=0.001, patience=patience)

    # train model
    history = model.fit(X_train,
                        y_train,
                        epochs=epochs,
                        batch_size=batch_size,
                        validation_data=(X_validation, y_validation),
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
    # generate train, validation and test sets
    X_train, y_train, X_validation, y_validation, X_test, y_test = prepare_dataset(DATA_PATH)

    # create network
    # (# of segments(or time window)(total amount of sample / hop_length), # of coefficients(13), depth)  )
    input_shape = (X_train.shape[1], X_train.shape[2], 1) 
    model = build_model(input_shape, learning_rate=LEARNING_RATE)

    # train network
    history = train(model, EPOCHS, BATCH_SIZE, PATIENCE, X_train, y_train, X_validation, y_validation)
    # model.fit(X_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE,
    #           validation_data=(X_validation, y_validation)
        
    # plot accuracy/loss for training/validation set as a function of the epochs
    plot_history(history)

    # evaluate network on test set
    start_time = timeit.default_timer()
    test_loss, test_acc = model.evaluate(X_test, y_test)
    terminate_time = timeit.default_timer()
    latency = (terminate_time - start_time) / X_test.shape[0] * 1000
    print("\nTest loss: {}, test accuracy: {}".format(test_loss, 100*test_acc))

    print("\nlatency time: {}ms".format(latency))
    # save model
    model.save(SAVED_MODEL_PATH)


if __name__ == "__main__":
    main()
