#!/usr/bin/env python
"""
Classify music clips according to genre.

Usage:
    After completing the steps listed in the README file, use this command:
    ./rnn2.py -e 16 -w /home/ubuntu/nervana/music -r 0 -s rnn2.pkl -v
"""

from neon.util.argparser import NeonArgparser
from neon.initializers import Gaussian, GlorotUniform
from neon.layers import Conv, Pooling, GeneralizedCost, Affine, LSTM, RecurrentMean
from neon.optimizers import Adagrad
from neon.transforms import Logistic, Tanh, Rectlin, Softmax, CrossEntropyMulti, Misclassification
from neon.models import Model
from neon.data import DataLoader, AudioParams
from neon.callbacks.callbacks import Callbacks
from util import create_index_files, display


parser = NeonArgparser(__doc__)
args = parser.parse_args()
train_idx, valid_idx = create_index_files(args.data_dir)

common_params = dict(sampling_freq=22050, clip_duration=16000, frame_duration=16)
train_params = AudioParams(**common_params)
valid_params = AudioParams(**common_params)
common = dict(target_size=1, nclasses=10, repo_dir=args.data_dir)
train = DataLoader(set_name='music-train', media_params=train_params,
                   index_file=train_idx, shuffle=True, **common)
valid = DataLoader(set_name='music-valid', media_params=valid_params,
                   index_file=valid_idx, shuffle=False, **common)
init = Gaussian(scale=0.01)
layers = [Conv((2, 2, 4), init=init, activation=Rectlin(),
               strides=dict(str_h=2, str_w=4)),
          Pooling(2, strides=2),
          Conv((3, 3, 4), init=init, batch_norm=True, activation=Rectlin(),
               strides=dict(str_h=1, str_w=2)),
          LSTM(128, init=GlorotUniform(), gate_activation=Tanh(),
               activation=Logistic(), reset_cells=True),
          RecurrentMean(),
          Affine(nout=common['nclasses'], init=init, activation=Softmax())]

model = Model(layers=layers)
opt = Adagrad(learning_rate=0.01, gradient_clip_value=15)
metric = Misclassification()
callbacks = Callbacks(model, eval_set=valid, metric=metric, **args.callback_args)
cost = GeneralizedCost(costfunc=CrossEntropyMulti())

model.fit(train, optimizer=opt, num_epochs=args.epochs, cost=cost, callbacks=callbacks)
print('Misclassification error = %.1f%%' % (model.eval(valid, metric=metric)*100))
display(model, ['Convolution_0'], 'inputs')
display(model, ['Convolution_0', 'Convolution_1', 'Pooling_0'], 'outputs')
