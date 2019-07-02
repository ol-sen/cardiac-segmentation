#!/usr/bin/env python

from __future__ import division, print_function

import os
import argparse
import logging
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np

from keras import losses, optimizers, utils
from keras.optimizers import SGD, RMSprop, Adagrad, Adadelta, Adam, Adamax, Nadam
from keras import callbacks
from keras.callbacks import ModelCheckpoint

from keras import backend as K
from keras.utils import multi_gpu_model

from sklearn.model_selection import KFold

from rvseg import dataset, models, loss, opts

def select_optimizer(optimizer_name, optimizer_args):
    optimizers = {
        'sgd': SGD,
        'rmsprop': RMSprop,
        'adagrad': Adagrad,
        'adadelta': Adadelta,
        'adam': Adam,
        'adamax': Adamax,
        'nadam': Nadam,
    }
    if optimizer_name not in optimizers:
        raise Exception("Unknown optimizer ({}).".format(name))

    return optimizers[optimizer_name](**optimizer_args)

def save_plot(figname, history):
    # "Accuracy"
    plt.figure(figsize=(12, 3.75))
    plt.subplot(1, 2, 1)
    plt.title('Dice')
    plt.plot(history.history['dice'])
    plt.plot(history.history['val_dice'])
    plt.ylabel('dice')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.show()
    # "Loss"
    plt.subplot(1, 2, 2)
    plt.title('Loss')
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.show()
    plt.savefig(figname, bbox_inches='tight')

class MultiGPUModelCheckpoint(callbacks.Callback):

    def __init__(self, filepath, base_model, monitor='val_loss', verbose=0,
                 save_best_only=False, save_weights_only=False,
                 mode='auto', period=1):
        super(MultiGPUModelCheckpoint, self).__init__()
        self.base_model = base_model
        self.monitor = monitor
        self.verbose = verbose
        self.filepath = filepath
        self.save_best_only = save_best_only
        self.save_weights_only = save_weights_only
        self.period = period
        self.epochs_since_last_save = 0

        if mode not in ['auto', 'min', 'max']:
            warnings.warn('ModelCheckpoint mode %s is unknown, '
                          'fallback to auto mode.' % (mode),
                          RuntimeWarning)
            mode = 'auto'

        if mode == 'min':
            self.monitor_op = np.less
            self.best = np.Inf
        elif mode == 'max':
            self.monitor_op = np.greater
            self.best = -np.Inf
        else:
            if 'acc' in self.monitor or self.monitor.startswith('fmeasure'):
                self.monitor_op = np.greater
                self.best = -np.Inf
            else:
                self.monitor_op = np.less
                self.best = np.Inf

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        self.epochs_since_last_save += 1
        if self.epochs_since_last_save >= self.period:
            self.epochs_since_last_save = 0
            filepath = self.filepath.format(epoch=epoch + 1, **logs)
            if self.save_best_only:
                current = logs.get(self.monitor)
                if current is None:
                    warnings.warn('Can save best model only with %s available, '
                                  'skipping.' % (self.monitor), RuntimeWarning)
                else:
                    if self.monitor_op(current, self.best):
                        if self.verbose > 0:
                            print('Epoch %05d: %s improved from %0.5f to %0.5f,'
                                  ' saving model to %s'
                                  % (epoch + 1, self.monitor, self.best,
                                     current, filepath))
                        self.best = current
                        if self.save_weights_only:
                            self.base_model.save_weights(filepath, overwrite=True)
                        else:
                            self.base_model.save(filepath, overwrite=True)
                    else:
                        if self.verbose > 0:
                            print('Epoch %05d: %s did not improve' %
                                  (epoch + 1, self.monitor))
            else:
                if self.verbose > 0:
                    print('Epoch %05d: saving model to %s' % (epoch + 1, filepath))
                if self.save_weights_only:
                    self.base_model.save_weights(filepath, overwrite=True)
                else:
                    self.base_model.save(filepath, overwrite=True)

def train():
    logging.basicConfig(level=logging.INFO)

    args = opts.parse_arguments()

    logging.info("Loading dataset...")
    augmentation_args = {
        'rotation_range': args.rotation_range,
        'width_shift_range': args.width_shift_range,
        'height_shift_range': args.height_shift_range,
        'shear_range': args.shear_range,
        'zoom_range': args.zoom_range,
        'fill_mode' : args.fill_mode,
        'alpha': args.alpha,
        'sigma': args.sigma,
    }

    images, masks = dataset.load_images(args.datadir, args.classes)
    # get image dimensions
    _, height, width, channels = images[0].shape
    _, _, _, classes = masks[0].shape
    
    logging.info("Building model...")
    string_to_model = {
        "unet": models.unet,
        "dilated-unet": models.dilated_unet,
        "dilated-densenet": models.dilated_densenet,
        "dilated-densenet2": models.dilated_densenet2,
        "dilated-densenet3": models.dilated_densenet3,
    }

    if args.multi_gpu:
         with tf.device('/cpu:0'):
            model = string_to_model[args.model]
            m = model(height=height, width=width, channels=channels,
                classes=classes, features=args.features, depth=args.depth, 
                padding=args.padding, temperature=args.temperature, 
                batchnorm=args.batchnorm, dropout=args.dropout)

            m.summary()

            if args.load_weights:
                logging.info("Loading saved weights from file: {}".format(args.load_weights))
                m.load_weights(args.load_weights)
    else:
        model = string_to_model[args.model]
        m = model(height=height, width=width, channels=channels,
            classes=classes, features=args.features, depth=args.depth, 
            padding=args.padding, temperature=args.temperature, 
            batchnorm=args.batchnorm, dropout=args.dropout)

        m.summary()

        if args.load_weights:
            logging.info("Loading saved weights from file: {}".format(args.load_weights))
            m.load_weights(args.load_weights)

    # instantiate optimizer, and only keep args that have been set
    # (not all optimizers have args like `momentum' or `decay')
    optimizer_args = {
        'lr':       args.learning_rate,
        'momentum': args.momentum,
        'decay':    args.decay
    }
    for k in list(optimizer_args):
        if optimizer_args[k] is None:
            del optimizer_args[k]
    optimizer = select_optimizer(args.optimizer, optimizer_args)

    # select loss function: pixel-wise crossentropy, soft dice or soft
    # jaccard coefficient
    if args.loss == 'pixel':
        def lossfunc(y_true, y_pred):
            return loss.weighted_categorical_crossentropy(
                y_true, y_pred, args.loss_weights)
    elif args.loss == 'dice':
        def lossfunc(y_true, y_pred):
            return loss.sorensen_dice_loss(y_true, y_pred, args.loss_weights)
    elif args.loss == 'jaccard':
        def lossfunc(y_true, y_pred):
            return loss.jaccard_loss(y_true, y_pred, args.loss_weights)
    else:
        raise Exception("Unknown loss ({})".format(args.loss))

    def dice(y_true, y_pred):
        batch_dice_coefs = loss.sorensen_dice(y_true, y_pred, axis=[1, 2])
        dice_coefs = K.mean(batch_dice_coefs, axis=0)
        return dice_coefs[1]    # HACK for 2-class case

    def jaccard(y_true, y_pred):
        batch_jaccard_coefs = loss.jaccard(y_true, y_pred, axis=[1, 2])
        jaccard_coefs = K.mean(batch_jaccard_coefs, axis=0)
        return jaccard_coefs[1] # HACK for 2-class case

    metrics = ['accuracy', dice, jaccard]
    
    if args.multi_gpu:
        parallel_model = multi_gpu_model(m, gpus=2)
        parallel_model.compile(optimizer=optimizer, loss=lossfunc, metrics=metrics)
    else:
        m.compile(optimizer=optimizer, loss=lossfunc, metrics=metrics)
    
    train_indexes = []
    val_indexes = []
    if args.cross_val_folds is not None:
        if args.cross_val_folds > len(images):
            raise Exception("Number of cross validation folds must be not more than {}.".format(len(images)))

        kf = KFold(n_splits = 4)
        val_dice_values = []
        fold = 1
        for train_indexes, val_indexes in kf.split(images):
            print("fold #{}".format(fold))
            print("{} {}".format(train_indexes, val_indexes))
            train_generator, train_steps_per_epoch, \
                val_generator, val_steps_per_epoch = dataset.create_generators(
                    args.datadir, args.batch_size,
                    train_indexes, val_indexes,
                    validation_split=args.validation_split,
                    mask=args.classes,
                    shuffle_train_val=args.shuffle_train_val,
                    shuffle=args.shuffle,
                    seed=args.seed,
                    normalize_images=args.normalize,
                    augment_training=args.augment_training,
                    augment_validation=args.augment_validation,
                    augmentation_args=augmentation_args)
        
            # automatic saving of model during training
            if args.checkpoint:
                monitor = 'val_dice'
                mode = 'max'
                filepath = os.path.join(
                        args.outdir, "weights-{epoch:02d}-{val_dice:.4f}" + "-fold{}.hdf5".format(fold)) 
        
                if args.multi_gpu:
                    checkpoint = MultiGPUModelCheckpoint(
                        filepath, m, monitor=monitor, verbose=1,
                        save_best_only=True, mode=mode)
                else:
                    checkpoint = ModelCheckpoint(
                        filepath, monitor=monitor, verbose=1,
                        save_best_only=True, mode=mode)

                callbacks = [checkpoint]
            else:
                callbacks = []

            # train
            logging.info("Begin training.")
            if args.multi_gpu:
                history = parallel_model.fit_generator(train_generator,
                    epochs=args.epochs,
                    steps_per_epoch=train_steps_per_epoch,
                    validation_data=val_generator,
                    validation_steps=val_steps_per_epoch,
                    callbacks=callbacks,
                    verbose=2)
            else:
                history = m.fit_generator(train_generator,
                    epochs=args.epochs,
                    steps_per_epoch=train_steps_per_epoch,
                    validation_data=val_generator,
                    validation_steps=val_steps_per_epoch,
                    callbacks=callbacks,
                    verbose=2)

            save_plot(args.outfile_plot + "-fold{}.png".format(fold), history)
            m.save(os.path.join(args.outdir, args.outfile + "-fold{}.hdf5".format(fold)))
            val_dice_values += [max(history.history['val_dice'])]
            fold += 1

        print("Maximum dice values on validation sets are {}.".format(val_dice_values))
        print("Mean dice value is {}.".format(np.mean(val_dice_values)))
    else:
        train_generator, train_steps_per_epoch, \
            val_generator, val_steps_per_epoch = dataset.create_generators(
                args.datadir, args.batch_size,
                train_indexes, val_indexes,
                validation_split=args.validation_split,
                mask=args.classes,
                shuffle_train_val=args.shuffle_train_val,
                shuffle=args.shuffle,
                seed=args.seed,
                normalize_images=args.normalize,
                augment_training=args.augment_training,
                augment_validation=args.augment_validation,
                augmentation_args=augmentation_args)

        # automatic saving of model during training
        if args.checkpoint:
            monitor = 'val_dice'
            mode = 'max'
            filepath = os.path.join(
                args.outdir, "weights-{epoch:02d}-{val_dice:.4f}.hdf5") 
        
            if args.multi_gpu:
                checkpoint = MultiGPUModelCheckpoint(
                    filepath, m, monitor=monitor, verbose=1,
                    save_best_only=True, mode=mode)
            else:
                checkpoint = ModelCheckpoint(
                    filepath, monitor=monitor, verbose=1,
                    save_best_only=True, mode=mode)

            callbacks = [checkpoint]
        else:
            callbacks = []
    
        # train
        logging.info("Begin training.")
        if args.multi_gpu:
            history = parallel_model.fit_generator(train_generator,
                epochs=args.epochs,
                steps_per_epoch=train_steps_per_epoch,
                validation_data=val_generator,
                validation_steps=val_steps_per_epoch,
                callbacks=callbacks,
                verbose=2)
        else:
            history = m.fit_generator(train_generator,
                epochs=args.epochs,
                steps_per_epoch=train_steps_per_epoch,
                validation_data=val_generator,
                validation_steps=val_steps_per_epoch,
                callbacks=callbacks,
                verbose=2)

        save_plot(args.outfile_plot + ".png", history)
        m.save(os.path.join(args.outdir, args.outfile + ".hdf5"))

        print("Maximum dice value on validation set is {}.".format(max(history.history['val_dice'])))

if __name__ == '__main__':
    
    train()

K.clear_session()
