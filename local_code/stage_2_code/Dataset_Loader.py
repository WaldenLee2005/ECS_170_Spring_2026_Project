'''
Concrete IO class for a specific dataset
'''

# Copyright (c) 2017-Current Jiawei Zhang <jiawei@ifmlab.org>
# License: TBD

from local_code.base_class.dataset import dataset
import pandas as pd
import numpy as np

class Dataset_Loader(dataset):
    train_file_path = None
    test_file_path = None

    def __init__(self, dName=None, dDescription=None):
        super().__init__(dName, dDescription)

    def load(self):
        print('loading train/test data...')

        train_df = pd.read_csv(self.train_file_path, header=None)
        test_df = pd.read_csv(self.test_file_path, header=None)

        X_train = train_df.iloc[:, 1:].values.astype(np.float32)
        y_train = train_df.iloc[:, 0].values.astype(np.int64)

        X_test = test_df.iloc[:, 1:].values.astype(np.float32)
        y_test = test_df.iloc[:, 0].values.astype(np.int64)

        # normalize
        X_train = X_train / 255.0
        X_test = X_test / 255.0

        print('train X shape:', X_train.shape)
        print('train y shape:', y_train.shape)
        print('test X shape:', X_test.shape)
        print('test y shape:', y_test.shape)

        return {
            'train': {'X': X_train, 'y': y_train},
            'test': {'X': X_test, 'y': y_test}
        }