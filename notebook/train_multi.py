import sys
import numpy as np
from classify_methods import *
import keras

def run(cycles):
    image_width = 128
    image_height = 96
    dense_param = 64
    threshold = 6000
    batch_size = 32
    seed = 42
    # FIXME: add test on base_fn + h5 to exit early
    multi = '+'.join([str(i) for i in cycles])
    base_fn = "model_v5_{}_{}_dense{}_threshold{}_batchsize{}".format(
         multi, image_width, dense_param, threshold, batch_size)
            
    model = get_model(image_width, image_height, dense_param)
                
    log_dir = '/notebooks/logs/' + base_fn
    tensorboard = keras.callbacks.TensorBoard(
                    log_dir=log_dir, histogram_freq=1, batch_size=batch_size,
                    write_graph=True, write_grads=False, write_images=False,
                    embeddings_freq=None, embeddings_layer_names=None, embeddings_metadata=None)
    X = None
    y = []
    for index, cycle in enumerate(cycles):
        _X, _y = get_dataset(cycle, image_width, image_height, threshold)
        if X is None:
            X = _X
        else:
            X = np.append(X, _X, axis=0)
        y.extend(_y)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed)
    Y_train = np_utils.to_categorical(y_train)
    Y_test = np_utils.to_categorical(y_test)
    cls_weight = class_weight.compute_class_weight('balanced', np.unique(y_train), y_train)

    hst = model.fit(X_train, Y_train, epochs=20, verbose=1, batch_size=batch_size, 
                    validation_data=(X_test, Y_test), class_weight=cls_weight,
                    callbacks=[tensorboard])
    
    model.save('data/' + base_fn + '-{}-{}'.format(index, cycle) + '.h5')
    print("saved" + base_fn)
    save_scores(base_fn + '-{}-{}'.format(index, cycle), model.evaluate(X_test, Y_test, verbose=0), model.evaluate(X_train, Y_train, verbose=0))

# cress seramis:
#run([71, 72, 74, 78, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109])
# phacelia seramis: (too big for the ram of my machine)
#run([64, 65, 66, 67, 68, 69, 70, 73, 75, 76, 77, 83, 84, 87, 88, 89, 90, 91, 92, 94, 95, 96, 97])
# phacelia seramis: (the part used)
run([68, 70, 75, 77, 83, 84, 90, 94, 96])
                    
