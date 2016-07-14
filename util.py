"""
Utility functions for pattern recognition examples
"""

import os
import glob
import numpy as np
import matplotlib
matplotlib.use('agg')
from matplotlib import pyplot as plt


def create_index_files(source_path, train_percent=90, pattern='*'):
    assert os.path.exists(source_path)
    train_idx = os.path.join(source_path, os.pardir, 'music-train-index.csv')
    valid_idx = os.path.join(source_path, os.pardir, 'music-valid-index.csv')
    if os.path.exists(train_idx) and os.path.exists(valid_idx):
        return train_idx, valid_idx
    subdirs = glob.iglob(os.path.join(source_path, '*'))
    subdirs = list(filter(lambda x: os.path.isdir(x), subdirs))
    classes = sorted(map(lambda x: os.path.basename(x), subdirs))
    class_map = {key: val for key, val in zip(classes, range(len(classes)))}

    # Split into training and validation subsets.
    np.random.seed(0)
    with open(train_idx, 'w') as train_fd, open(valid_idx, 'w') as valid_fd:
        train_fd.write('filename,label\n')
        valid_fd.write('filename,label\n')
        for subdir in subdirs:
            label = class_map[os.path.basename(subdir)]
            files = glob.glob(os.path.join(subdir, pattern))
            np.random.shuffle(files)
            train_count = (len(files) * train_percent) // 100
            for idx, filename in enumerate(files):
                fd = train_fd if idx < train_count else valid_fd
                rel_path = os.path.join(os.path.basename(subdir),
                                        os.path.basename(filename))
                fd.write(rel_path + ',' + str(label) + '\n')
    return train_idx, valid_idx


def display(model, layer_names, sel):
    layers = model.layers.layers
    for layer in layers:
        if layer.name not in layer_names:
            continue
        data = getattr(layer, sel)
        shape = layer.in_shape if sel == 'inputs' else layer.out_shape
        for i in [12, 84]:
            imgs = data[:, i].get().reshape(shape)
            if len(imgs.shape) < 3:
                continue
            for j in range(imgs.shape[0]):
                path = os.path.join('imgs', str(i))
                if not os.path.exists(path):
                    os.makedirs(path)
                name = os.path.join(path, layer.name + '.' + sel + '.' +
                                    str(j) + '.png')
                plt.imshow(imgs[j])
                plt.savefig(name, bbox_inches='tight')
