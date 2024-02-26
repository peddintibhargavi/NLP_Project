import os
import librosa
import numpy as np
import tensorflow as tf
from pickledump import PickleDumpLoad
from warnings import simplefilter
from tensorflow.keras.layers import Input, Conv1D, MaxPooling1D, Dropout, Dense, LSTM, Flatten
from tensorflow.keras.models import Model
from datasets import Datasets
from traindatagenerator import DataGenerator

simplefilter(action='ignore', category=FutureWarning)

class ModelTest(object):

    def __init__(self):
        # Test path
        self.TEST_PATH = './datasets/test'

        # Load the saved model
        MODEL_SAVED_PATH = "./model/model.h5"
        self.model = tf.keras.models.load_model(MODEL_SAVED_PATH)

        # Load the label model
        pickledump = PickleDumpLoad()
        self.label_model = pickledump.load_config('label.mdl')

    def load_sound(self, path):
        time_series_x, sampling_rate = librosa.load(path, sr=None, mono=True)

        # Extract mfcc
        mfccs = librosa.feature.mfcc(y=time_series_x, sr=sampling_rate, n_mfcc=20)

        # Extract melspectogram
        mel = librosa.feature.melspectrogram(y=time_series_x, sr=sampling_rate, n_mels=20,
                                              fmax=8000, win_length=1024, hop_length=320)

        mfccs_scaled_features = np.mean(mfccs.T, axis=0)
        mel_s_scaled_features = np.mean(mel.T, axis=0)

        # Multiply the mfcc and melspectogram: for additional features
        multiply = np.multiply(mel_s_scaled_features, mfccs_scaled_features)

        mfccs = tf.convert_to_tensor([mfccs_scaled_features, multiply, mel_s_scaled_features])
        mfccs = tf.reshape(mfccs, shape=[1, 20, 3])

        return mfccs

    def check_prediction(self, pred):
        # Get the based label
        label_model = self.label_model

        # Get the prediction
        predicted = tf.argmax(pred, axis=1)

        # Decode the label
        result = None
        try:
            result = label_model[predicted.numpy()[0]]
        except:
            pass

        return result

    def predict(self, path):
        # Load sound
        x_test = self.load_sound(path)

        # Make a prediction
        pred = self.model.predict(x_test)

        # Check the prediction
        result = self.check_prediction(pred)

        return result

    def main(self):
        test_paths = os.listdir(self.TEST_PATH)
        for x in test_paths:
            path = os.path.join(self.TEST_PATH, x)
            # Make a prediction
            result = self.predict(path)
            # Print result
            print(f'{path} = {result}')


class ModelTrain(object):

    def __init__(self):
        self.datasets = Datasets()
        self.PATH = "./datasets/train"
        self.MODEL_SAVED_PATH = "./model/model.h5"
        self.EPOCHS = 15
        self.BATCH_SIZE = 2

    def define_model(self, CLASSES=2, INPUT_SHAPE=[20, 3]):
        input = Input(shape=INPUT_SHAPE)

        m = Conv1D(32, 3, activation='relu')(input)
        m = Conv1D(64, 3, activation='relu')(m)
        m = MaxPooling1D()(m)
        m = Dropout(0.25)(m)
        m = Dense(128, activation='relu')(m)
        m = LSTM(128, return_sequences=True)(m)
        m = LSTM(128, return_sequences=False)(m)
        m = Dense(256, activation='relu')(m)
        m = Flatten()(m)
        m = Dense(128, activation='relu')(m)
        m = Dense(CLASSES, activation='sigmoid')(m)

        model = Model(input, m)

        model.summary()

        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss=tf.keras.losses.CategoricalCrossentropy(),
            metrics=['mse', 'accuracy'],
        )

        return model

    def train(self):
        # Get all the train data paths
        x_paths, x_label = self.datasets.get_data_paths(self.PATH)

        # Apply one hot encoder to the data labels
        x_label = self.datasets.one_hot_encoder(x_label=x_label)

        # Get the total class and set the data input shape
        INPUT_SHAPE = [20, 3]
        CLASSES = len(x_label[0])

        # Prepare the training data
        x_y_train = DataGenerator(
            x_train=x_paths,
            y_train=x_label,
            batch_size=self.BATCH_SIZE)

        # Prepare the model
        model = self.define_model(CLASSES=CLASSES, INPUT_SHAPE=INPUT_SHAPE)

        # Train the model
        history = model.fit(
            x_y_train,
            epochs=self.EPOCHS,
            batch_size=self.BATCH_SIZE,
            # callbacks=tf.keras.callbacks.EarlyStopping(verbose=1, patience=2),
        )

        # Save the model
        model.save(self.MODEL_SAVED_PATH)


class DataGenerator(tf.keras.utils.Sequence):

    def __init__(self, x_train, y_train, batch_size=4, shuffle=True, frame_length=20):
        self.batch_size = batch_size
        self.x_train = x_train
        self.y_train = y_train
        self.shuffle = shuffle
        self.frame_length = frame_length
        self.on_epoch_end()

    def __len__(self):
        return int(np.floor(len(self.y_train) / self.batch_size))

    def __getitem__(self, index):
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]

        x_train = []
        for path in np.array(self.x_train)[indexes]:
            data = self.data_load(path)
            x_train.append(data)

        x_train = np.array(x_train)
        y_train = np.array(self.y_train)[indexes]

        return x_train, y_train

    def on_epoch_end(self):
        self.indexes = np.arange(len(self.y_train))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def data_load(self, path):
        time_series_x, sampling_rate = librosa.load(path, sr=None, mono=True)

        # Extract mfcc
        mfccs = librosa.feature.mfcc(y=time_series_x, sr=sampling_rate, n_mfcc=20)

        # Extract melspectogram
        mel = librosa.feature.melspectrogram(y=time_series_x, sr=sampling_rate, n_mels=20,
                                              fmax=8000, win_length=1024, hop_length=320)

        mfccs_scaled_features = np.mean(mfccs.T, axis=0)
        mel_s_scaled_features = np.mean(mel.T, axis=0)

        # Multiply the mfcc and melspectogram: for additional features
        multiply = np.multiply(mel_s_scaled_features, mfccs_scaled_features)

        mfccs = tf.convert_to_tensor([mfccs_scaled_features, multiply, mel_s_scaled_features])
        mfccs = tf.reshape(mfccs, shape=[20, 3])

        return mfccs


if __name__ == '__main__':
    app_train = ModelTrain()
    app_train.train()

    app_test = ModelTest()
    app_test.main()
