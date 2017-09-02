import sys
from classify_methods import *
import keras


def run(cycle):
    image_width = 128
    image_height = 96
    dense_param = 64
    threshold = 8000
    batch_size = 32
    seed = 42
    X_train, X_test, Y_train, Y_test, cls_weight = render_dataset(
               cycle, image_width, image_height, threshold, seed)
    # FIXME: add test on base_fn + h5 to exit early
    base_fn = "model_v3_{}_{}_dense{}_threshold{}_batchsize{}".format(
         cycle, image_width, dense_param, threshold, batch_size)

    model = get_model(image_width, image_height, dense_param)

    log_dir = '/notebooks/logs/' + base_fn
    tensorboard = keras.callbacks.TensorBoard(
                    log_dir=log_dir, histogram_freq=1, batch_size=batch_size,
                    write_graph=True, write_grads=False, write_images=False,
                    embeddings_freq=None, embeddings_layer_names=None, embeddings_metadata=None)

    hst = model.fit(X_train, Y_train, epochs=20, verbose=1, batch_size=batch_size,
                    validation_data=(X_test, Y_test), class_weight=cls_weight,
                   callbacks=[tensorboard])

    model.save('data/' + base_fn + '.h5')
    print("saved" + base_fn)
    save_scores(base_fn, model.evaluate(X_test, Y_test, verbose=0), model.evaluate(X_train, Y_train, verbose=0))


run(int(sys.argv[1]))
