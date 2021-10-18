import argparse
import os
import numpy as np
import pandas as pd
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras import layers
from tensorflow.keras.utils import Sequence
from sklearn.preprocessing import MinMaxScaler
import time
import tensorflow as tf


parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int, default=64, help='Batch Size during training [default: 128]')
parser.add_argument('--max_epoch', type=int, default=100, help='Epoch to run [default: 50]')
parser.add_argument('--step_per_epoch', type=int, default=200, help='Step per epoch [default: 50]')
parser.add_argument('--num_parameter', type=int, default=4, help='# of parameter [default: 5]')
parser.add_argument('--optimizer', default='adam', help='adam or momentum [default: adam]')
parser.add_argument('--sliding_window', type=int, default=100, help='# of sliding window [default: 300]')
parser.add_argument('--num_output', type=int, default=2, help='# of output [default: 2]')
parser.add_argument('--log_dir', default='log', help='Log dir [default: log]')
parser.add_argument('--model_path', default='', help='model file path. ex)./my_model.h5 [default:]')
FLAGS = parser.parse_args()

BATCH_SIZE = FLAGS.batch_size
MAX_EPOCH = FLAGS.max_epoch
STEP_PER_EPOCH = FLAGS.step_per_epoch
OPTIMIZER = FLAGS.optimizer
NUM_PARAMETER = FLAGS.num_parameter
SLIDING_WINDOW = FLAGS.sliding_window
NUM_OUTPUT = FLAGS.num_output
LOG_DIR = FLAGS.log_dir
MODEL_PATH = FLAGS.model_path

LIMITED_SPPED = [1, 2]
DIRECTION = "PLEFT"   # "ALL", "LEFT", "RIGHT', "PLEFT", "PRIGHT"
SUPPORT_PARAMETER = 'TorsoVelocity'  # ['TorsoAngle', 'TorsoVelocity']
EARLY_STOP_PATIENCE = MAX_EPOCH // 10


TRAIN_FILES = './datasets/train_indices.csv'
EVALUATE_FILES = './datasets/test_indices.csv'


def make_folder(time, root=""):
    path = root + "%04d-%02d-%02d_%02d_%02d_%02d" % (
        time.tm_year, time.tm_mon, time.tm_mday, time.tm_hour, time.tm_min, time.tm_sec)
    if not os.path.isdir(path):
        os.mkdir(path)
        return path



class DataGenerator(Sequence):
    def __init__(self, path, list_IDs, batch_size, cycle_length, parameter_size, direction, num_classes):
        self.path = path
        self.list_IDs = list_IDs
        self.dataLength = self.SearchMaxLength()
        self.len_list_path = len(list_IDs)
        self.batch_size = batch_size
        self.cycle_length = cycle_length
        self.parameter_size = parameter_size
        self.num_classes = num_classes
        self.direction = direction
        self.indexes = np.arange(len(self.list_IDs))

    # 실제 코드용 모든 데이터 활용
    def __len__(self):
        len_ = int(self.dataLength/self.batch_size)
        if len_ * self.batch_size < self.dataLength:
            len_ += 1
        return len_

    # # 개발용. 적은 수의 iterator
    # def __len__(self):
    #     len_ = int(len(self.list_IDs)/self.batch_size)
    #     if len_ * self.batch_size < len(self.list_IDs):
    #         len_ += 1
    #     return len_

    def __getitem__(self, index):
        X_data = np.zeros((self.batch_size, self.cycle_length, self.parameter_size), dtype=np.float32)
        y_label = np.zeros((self.batch_size, self.cycle_length, self.num_classes), dtype=np.float32)
        i = 0
        while i < self.batch_size:
            indexes = np.random.randint(0, self.len_list_path-1)
            data_path = self.path + self.list_IDs[indexes]

            filename = os.path.basename(data_path)
            subject_info = filename.split('_')
            subject_speed = int(subject_info[2][1])

            if subject_speed in LIMITED_SPPED:
                continue

            data_raw = pd.read_csv(data_path, header=0)
            length_data = len(data_raw['Index'])    # length_data > cycle_length 아니면 다시 랜덤하게 선정
            data_raw = data_raw.drop(columns=['Index'], axis=1)
            if length_data <= self.cycle_length:
                # print("{}<{}".format(length_data, self.cycle_length))
                continue
            else:
                X_data[i, ], y_label[i, ] = self.__data_generation(data_raw, length_data)
                i += 1

        return X_data, y_label

    def SearchMaxLength(self):
        total_size = 0
        df_list = []
        for idx, _path in enumerate(self.list_IDs):
            df = pd.read_csv(self.path + _path)

            filename = os.path.basename(_path)
            subject_info = filename.split('_')
            subject_speed = int(subject_info[2][1])

            if subject_speed in LIMITED_SPPED:
                continue
            df_list.append(df)
            df_size = len(df)
            if df_size > SLIDING_WINDOW:
                total_size += (df_size - SLIDING_WINDOW)
        result = df_list[0]
        for i in range(1, len(df_list)):
            result = pd.concat([result, df_list[i]])
        result = result.drop(columns=['Index', 'LeftHS', 'RightHS', 'PieceWiseLeft', 'PieceWiseRight'], axis=1)
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.scaler.fit(result)
        return total_size

    def __data_generation(self, data_raw, length):
        if length - self.cycle_length <= 0:
            print("{}<{}".format(length, self.cycle_length))
        start = np.random.randint(0, length - self.cycle_length)  # (0, length - self.cycle_length) 사이에서 랜덤값 선정

        label_raw = data_raw[['LeftHS', 'RightHS', 'PieceWiseLeft', 'PieceWiseRight']]
        data_raw = data_raw.drop(columns=['LeftHS', 'RightHS', 'PieceWiseLeft', 'PieceWiseRight'], axis=1)
        self.scaler.transform(data_raw)

        if self.parameter_size == 4:  # parameter size: 4
            params = ['ThighAngle', 'ThighVelocity', 'TorsoAngle', 'TorsoVelocity']
        elif self.parameter_size == 3:
            params = ['ThighAngle', 'ThighVelocity', SUPPORT_PARAMETER]
        else:  # parameter size: 2
            params = ['ThighAngle', 'ThighVelocity']
        labels = 'HS'
        if self.direction == "ALL":
            _direction = np.random.choice(["Left", "Right"])
        elif self.direction == "LEFT":
            _direction = "Left"
        elif self.direction == "RIGHT":
            _direction = "Right"
        elif self.direction == "PLEFT":
            _direction = "PLeft"
        elif self.direction == "PRIGHT":
            _direction = "PRight"
        else:
            _direction = ""
            print("direction ERROR")
        # left
        if _direction == "Left":
            params[0] = "L" + params[0]
            params[1] = "L" + params[1]
            labels = "Left" + labels
        elif _direction == "Right":
            params[0] = "R" + params[0]
            params[1] = "R" + params[1]
            labels = "Right" + labels
        elif _direction == "PLeft":
            params[0] = "L" + params[0]
            params[1] = "L" + params[1]
            labels = "PieceWiseLeft"
        elif _direction == "PLeft":
            params[0] = "R" + params[0]
            params[1] = "R" + params[1]
            labels = "PieceWiseRight"
        else:
            labels = ""

        data = data_raw[params][start:start+self.cycle_length].to_numpy()
        label_gait = label_raw[labels][start:start+self.cycle_length].to_numpy()
        if _direction[0] == "P":
            _sin = np.sin(label_gait * np.pi)
            _cos = np.cos(label_gait * np.pi)
        else:
            _sin = np.sin(label_gait * 2.0 * np.pi)
            _cos = np.cos(label_gait * 2.0 * np.pi)
        label = np.hstack((_sin.reshape((self.cycle_length, 1)),
                           _cos.reshape((self.cycle_length, 1))))

        return data, label

def create_model():
    xInput = layers.Input(shape=(SLIDING_WINDOW, NUM_PARAMETER))
    xLstm_1 = layers.LSTM(units=128, return_sequences=True)(xInput)
    xLstm_2 = layers.Bidirectional(layers.LSTM(units=64, return_sequences=True))(xLstm_1)
    xLstm_3 = layers.LSTM(units=64, return_sequences=True)(xLstm_2)
    xLstm_4 = layers.Bidirectional(layers.LSTM(units=32, return_sequences=True))(xLstm_3)
    xDropout = layers.Dropout(0.2)(xLstm_4)
    xOutput = layers.Dense(NUM_OUTPUT)(xDropout)
    model = tf.keras.Model(xInput, xOutput)
    model.compile(loss='mse', optimizer='adam')
    return model


def train_path_with_parameter(pathlist, cycle_length):
    result = []
    for _path in pathlist:
        filename = os.path.basename(_path)
        subject_info = filename.split('_')
        subject_speed = int(subject_info[2][1])

        if subject_speed in LIMITED_SPPED:
            continue

        _data = pd.read_csv("./datasets/piecewise/" + _path)
        len_data = len(_data)
        if len_data <= cycle_length:
            continue
        result.append(_path)

    return result


def train():
    train_path = pd.read_csv(TRAIN_FILES, header=None)[0]
    validation_path = pd.read_csv(EVALUATE_FILES, header=None)[0]

    train_ = train_path_with_parameter(train_path, SLIDING_WINDOW)
    validation_ = train_path_with_parameter(validation_path, SLIDING_WINDOW)

    train_generator = DataGenerator("./datasets/", train_,
                                    BATCH_SIZE, SLIDING_WINDOW, NUM_PARAMETER, DIRECTION, NUM_OUTPUT)

    validation_generator = DataGenerator("./datasets/", validation_,
                                         BATCH_SIZE, SLIDING_WINDOW, NUM_PARAMETER, DIRECTION, NUM_OUTPUT)

    model = create_model()
    print(model.summary())

    cb_ModelCheck = ModelCheckpoint(filepath='./checkpoint.keras', monitor='val_loss',
                                    verbose=1, save_weights_only=True, save_best_only=True)
    cb_EarlyStopping = EarlyStopping(monitor='val_loss', patience=EARLY_STOP_PATIENCE)

    cbs = [cb_ModelCheck, cb_EarlyStopping]
    model.fit(train_generator, steps_per_epoch=200, validation_data=validation_generator,
              epochs=MAX_EPOCH, verbose=2, validation_steps=50, callbacks=cbs)

    path = make_folder(time.localtime())

    SAVE_PATH = "./Frontiers_lstm_" + str(DIRECTION) + "_" + str(NUM_PARAMETER) + "_" + str(SLIDING_WINDOW) + "_wo"
    model.save(path + SAVE_PATH)

def GetScaler(path, list_IDs):
    df_list = []
    for idx, _path in enumerate(list_IDs):
        df = pd.read_csv(path + _path)

        filename = os.path.basename(_path)
        subject_info = filename.split('_')
        subject_speed = int(subject_info[2][1])

        if subject_speed in LIMITED_SPPED:
            continue
        df_list.append(df)

    result = df_list[0]

    for i in range(1, len(df_list)):
        result = pd.concat([result, df_list[i]])

    result = result.drop(columns=['Index', 'LeftHS', 'RightHS', 'PieceWiseLeft', 'PieceWiseRight'], axis=1)
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler.fit(result)
    return scaler


def DatasetFromFile(_path, _cycle, _parameter, _num_y, direction="PLEFT"):
        data_raw = pd.read_csv(_path, header=0)

        length = len(data_raw)
        if length - _cycle <= 0:
            print("{}<{}".format(length, _cycle))

        data = np.zeros((length - _cycle, _cycle, _parameter), dtype=np.float32)
        label = np.zeros((length - _cycle, _cycle, _num_y), dtype=np.float32)

        if _parameter == 4:  # parameter size: 4
            params = ['ThighAngle', 'ThighVelocity', 'TorsoAngle', 'TorsoVelocity']
        elif _parameter == 3:
            params = ['ThighAngle', 'ThighVelocity', SUPPORT_PARAMETER]
        else:  # parameter size: 2
            params = ['ThighAngle', 'ThighVelocity']

        labels = 'HS'
        if direction == "ALL":
            _direction = np.random.choice(["Left", "Right"])
        elif direction == "LEFT":
            _direction = "Left"
        elif direction == "RIGHT":
            _direction = "Right"
        elif direction == "PLEFT":
            _direction = "PLeft"
        elif direction == "PRIGHT":
            _direction = "PRight"
        else:
            _direction = ""
            print("direction ERROR")

        if _direction == "Left":
            params[0] = "L" + params[0]
            params[1] = "L" + params[1]
            labels = "Left" + labels
        elif _direction == "Right":
            params[0] = "R" + params[0]
            params[1] = "R" + params[1]
            labels = "Right" + labels
        elif _direction == "PLeft":
            params[0] = "L" + params[0]
            params[1] = "L" + params[1]
            labels = "PieceWiseLeft"
        elif _direction == "PLeft":
            params[0] = "R" + params[0]
            params[1] = "R" + params[1]
            labels = "PieceWiseRight"
        else:
            labels = ""

        for i in range(length - _cycle):
            data[i] = data_raw[params][i:i+_cycle].to_numpy()
            label_gait = data_raw[labels][i:i+_cycle].to_numpy()
            if _direction[0] == "P":
                label[i] = np.hstack((np.sin(label_gait * np.pi).reshape((_cycle, 1)),
                                      np.cos(label_gait * np.pi).reshape((_cycle, 1))))
            else:
                label[i] = np.hstack((np.sin(label_gait * 2.0 * np.pi).reshape((_cycle, 1)),
                                      np.cos(label_gait * 2.0 * np.pi).reshape((_cycle, 1))))

        return (data, label)


def predict():
    train_path = pd.read_csv(TRAIN_FILES, header=None)[0]
    validation_path = pd.read_csv(EVALUATE_FILES, header=None)[0]
    train_ = train_path_with_parameter(train_path, SLIDING_WINDOW)


    GetScaler("./datasets/piecewise/", train_)
    new_model = tf.keras.models.load_model(MODEL_PATH)

    for ix in range(len(validation_path)):
        _data = DatasetFromFile("./datasets/piecewise/"+validation_path[ix], SLIDING_WINDOW,
                                NUM_PARAMETER, NUM_OUTPUT, DIRECTION)

        y_pred = new_model.predict(_data[0])

        dir = "./result/"+os.path.splitext(os.path.basename(MODEL_PATH))[0]
        if not os.path.exists(dir):
            os.makedirs(dir)

        np.savetxt(dir+"/origin_" + validation_path[ix] + ".txt", _data[1][:, -1, :], fmt="%.8f")
        np.savetxt(dir+"/predict_" + validation_path[ix] + ".txt", y_pred[:, -1, :], fmt="%.8f")


if __name__ == "__main__":
    config = tf.compat.v1.ConfigProto(gpu_options=
                                      tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.8)
                                      # device_count = {'GPU': 1}
                                      )
    config.gpu_options.allow_growth = True
    session = tf.compat.v1.Session(config=config)
    tf.compat.v1.keras.backend.set_session(session)

    if MODEL_PATH == '':
        train()
    else:
        predict()