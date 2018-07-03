from models import *
from custom_metrics import metric_degrees_difference
from keras.utils import HDF5Matrix
from random_search import random_search
from kfold import test_model
import h5py
from google.colab import files
import urllib.request


h5file = 'processed_data_501.h5'
url = 'https://www.nikhef.nl/~pgunnink/Keras_train/test_data.h5'
response = urllib.request.urlretrieve(url, h5file)

traces_train = HDF5Matrix(h5file, 'traces')
labels_train = HDF5Matrix(h5file, 'labels')
input_features_train = HDF5Matrix(h5file, 'input_features')


fit_args = {
    'x':[traces_train, input_features_train],
    'y': labels_train,
    'batch_size': 2**10,
    'epochs' : 2,
    'shuffle': 'batch',
    'validation_split': 0.1,
    'verbose': 0}
A = Adam(lr=0.001)
compile_args= {
    'optimizer': A,
    'loss': 'mse',
    'metrics': [metric_degrees_difference]
}
model = baseModelDense
evaluation_params = {
    'model_fn': model,
    'fit_args': fit_args,
    'compile_args': compile_args,
    'k_fold': 2,
    'verbose': False
}
random_params = {
    'trace_filter_1': range(40,80),
    'trace_filter_2': range(30,60),
}
i = 0
while True:
    with h5py.File('result_%s.h5' % i) as f:
        result = random_search(test_model, evaluation_params, random_params)
        for key in result:
            f.create_dataset(key, data=result[key])
        print('Finished iteration %s' % i)

    files.download('result_%s.h5' % i)
    i += 1