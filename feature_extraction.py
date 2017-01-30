import pickle
import tensorflow as tf
# TODO: import Keras layers you need here
from keras.models import Sequential, Model
from keras.layers import Dense, Activation, Flatten, Input
#from keras.utils import np_utils
import numpy as np

flags = tf.app.flags
FLAGS = flags.FLAGS

# command line flags
flags.DEFINE_string('training_file', '', "Bottleneck features training file (.p)")
flags.DEFINE_string('validation_file', '', "Bottleneck features validation file (.p)")
flags.DEFINE_integer('batch_size',128,"size of batch (int)")
flags.DEFINE_integer('nb_epoch',50,"number of epoch (int)")



def load_bottleneck_data(training_file, validation_file):
    """
    Utility function to load bottleneck features.

    Arguments:
        training_file - String
        validation_file - String
    """
    print("Training file", training_file)
    print("Validation file", validation_file)

    with open(training_file, 'rb') as f:
        train_data = pickle.load(f)
    with open(validation_file, 'rb') as f:
        validation_data = pickle.load(f)

    X_train = train_data['features']
    y_train = train_data['labels']
    X_val = validation_data['features']
    y_val = validation_data['labels']

    return X_train, y_train, X_val, y_val


def main(_):
    # load bottleneck data
    X_train, y_train, X_val, y_val = load_bottleneck_data(FLAGS.training_file, FLAGS.validation_file)

    nb_classes = len(np.unique(y_train))
    #y_train_one_hot = np_utils.to_categorical(y_train,FLAGS.nb_classes)
    #y_val_one_hot = np_utils.to_categorical(y_val, FLAGS.nb_classes)

    print(X_train.shape, y_train.shape)
    print(X_val.shape, y_val.shape)

    # TODO: define your model and hyperparams here
    # make sure to adjust the number of classes based on
    # the dataset
    # 10 for cifar10
    # 43 for traffic

    model = Sequential()
    model.add(Flatten(input_shape=X_train.shape[1:]))
    #model.add(Dense(128))
    #model.add(Activation('relu'))
    model.add(Dense(nb_classes))
    model.add(Activation('softmax'))

    #input_shape = X_train.shape[1:]
    #inp = Input(shape=input_shape)
    #x = Flatten()(inp)
    #x = Dense(FLAGS.nb_classes, activation='softmax')(x)
    #model = Model(inp, x)


    # TODO: train your model here
    model.compile('adam','sparse_categorical_crossentropy',['accuracy'])
    history = model.fit(X_train,y_train,batch_size=FLAGS.batch_size,nb_epoch=FLAGS.nb_epoch,validation_data=(X_val,y_val))
    print("The validation accuracy is: %.3f.  It should be greater than 0.91" % history.history['val_acc'][-1])



# parses flags and calls the `main` function above
if __name__ == '__main__':
    tf.app.run()
