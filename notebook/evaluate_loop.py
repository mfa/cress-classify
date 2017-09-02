import sys
import glob
import os.path

from keras.models import load_model
import numpy as np

from classify_methods import *


def save_scores_cycle(fn, score_cycle, cycle, extra):
    scores = dict(extra)
    scores.update({
        'other_cycle': [cycle],
        'loss': [score_cycle[0]],
        'acc': [score_cycle[1]],
    })
    pandas.DataFrame.from_dict(scores).to_csv(fn, sep='|')


def evaluate(model_fn, other_cycle):
    print("=== {} --- {}".format(model_fn, other_cycle))
    # data/model_v3_101_128_dense32_threshold8000_batchsize32.h5
    parts = model_fn.split('.')[0].split('_')

    cycle = parts[2]
    image_width = int(parts[3])
    image_height = 96 * int(int(parts[3]) / 128)
    dense_param = int(parts[4][len('dense'):])
    threshold = int(parts[5][len('threshold'):])
    batch_size = int(parts[6].split('-')[0][len('batchsize'):])
    seed = 42
    
    csv_filename = model_fn.split('.')[0] + '--cycle--{}--scores.csv'.format(other_cycle)
    if os.path.exists(csv_filename):
        return
    
    model = load_model(model_fn)
    
    X, y = get_dataset(other_cycle, image_width, image_height, threshold, seed)
    Y = np_utils.to_categorical(y)
    extra = {
        'training_cycle': cycle,
        'image_width': image_width,
        'threshold': threshold,
        'dense_param': dense_param,
        'batch_size': batch_size
    }
    save_scores_cycle(csv_filename, model.evaluate(X, Y, verbose=0), other_cycle, extra)
    

for fn in glob.glob('data/*v5*.h5'):
    for cycle in sys.argv[1:]:
        evaluate(fn, int(cycle))
