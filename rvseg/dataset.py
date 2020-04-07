from __future__ import division, print_function

import numpy as np
from math import ceil
from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage.filters import gaussian_filter

from keras.preprocessing import image as keras_image
from keras.preprocessing.image import ImageDataGenerator

from . import RVSC

def random_elastic_deformation(image, alpha, sigma, mode='nearest',
                               random_state=None):
    """Elastic deformation of images as described in [Simard2003]_.
    .. [Simard2003] Simard, Steinkraus and Platt, "Best Practices for
       Convolutional Neural Networks applied to Visual Document Analysis", in
       Proc. of the International Conference on Document Analysis and
       Recognition, 2003.
    """
    assert len(image.shape) == 3

    if random_state is None:
        random_state = np.random.RandomState(None)

    height, width, channels = image.shape

    dx = gaussian_filter(2*random_state.rand(height, width) - 1,
                         sigma, mode="constant", cval=0) * alpha
    dy = gaussian_filter(2*random_state.rand(height, width) - 1,
                         sigma, mode="constant", cval=0) * alpha

    x, y = np.meshgrid(np.arange(height), np.arange(width), indexing='ij')
    indices = (np.repeat(np.ravel(x+dx), channels),
               np.repeat(np.ravel(y+dy), channels),
               np.tile(np.arange(channels), height*width))
    
    values = map_coordinates(image, indices, order=1, mode=mode)

    return values.reshape((height, width, channels))

class Iterator(object):
    def __init__(self, images, masks, batch_size, mode,
                 shuffle=True,
                 rotation_range=180,
                 width_shift_range=0.1,
                 height_shift_range=0.1,
                 shear_range=0.1,
                 zoom_range=0.01,
                 fill_mode='nearest',
                 alpha=500,
                 sigma=20):
        self.images = images
        self.masks = masks
        self.batch_size = batch_size
        self.mode = mode
        self.shuffle = shuffle
        augment_options = {
            'rotation_range': rotation_range,
            'width_shift_range': width_shift_range,
            'height_shift_range': height_shift_range,
            'shear_range': shear_range,
            'zoom_range': zoom_range,
            'fill_mode': fill_mode,
        }
        self.idg = ImageDataGenerator(**augment_options)
        self.alpha = alpha
        self.sigma = sigma
        self.fill_mode = fill_mode
        self.i = 0
        self.index = np.arange(len(images))
        if shuffle:
            np.random.shuffle(self.index)

    def __next__(self):
        return self.next()

    def next(self):
        # compute how many images to output in this batch
        start = self.i
        end = min(start + self.batch_size, len(self.images))
        
        augmented_images = []
        augmented_masks = []
        for m in self.index[start:end]:
            cur_patient_images = self.images[m]
            cur_patient_masks = self.masks[m]
            for n in range(len(cur_patient_images)):
                image = cur_patient_images[n]
                mask = cur_patient_masks[n]

                _, _, channels = image.shape

                # stack image + mask together to simultaneously augment
                stacked = np.concatenate((image, mask), axis=2)

                # apply simple affine transforms first using Keras
                augmented = self.idg.random_transform(stacked)

                # maybe apply elastic deformation
                if self.alpha != 0 and self.sigma != 0:
                    augmented = random_elastic_deformation(
                        augmented, self.alpha, self.sigma, self.fill_mode)

                # split image and mask back apart
                augmented_image = augmented[:,:,:channels]
                augmented_images.append(augmented_image)
                augmented_mask = np.round(augmented[:,:,channels:])
                augmented_masks.append(augmented_mask)

        self.i += self.batch_size
        if self.i >= len(self.images):
            self.i = 0
            if self.shuffle:
                np.random.shuffle(self.index)

        return np.asarray(augmented_images), np.asarray(augmented_masks)

class Generator(object):
    def __init__(self, images, masks, batch_size, mode, shuffle=True):
        self.images = images
        self.masks = masks
        self.batch_size = batch_size
        self.mode = mode
        self.shuffle = shuffle
        self.i = 0
        self.index = np.arange(len(images))
        if shuffle:
            np.random.shuffle(self.index)

    def __next__(self):
        return self.next()

    def next(self):
        # compute how many images to output in this batch
        start = self.i
        end = min(start + self.batch_size, len(self.images))

        images = []
        masks = []
        for m in self.index[start:end]: 
            cur_patient_images = self.images[m]
            cur_patient_images = self.images[m]
            cur_patient_masks = self.masks[m]

            for n in range(len(cur_patient_images)):
                image = cur_patient_images[n]
                mask = cur_patient_masks[n]
                images.append(image)
                masks.append(mask)

        self.i += self.batch_size
        if self.i >= len(self.images):
            self.i = 0
            if self.shuffle:
                np.random.shuffle(self.index)

        return np.asarray(images), np.asarray(masks)


def normalize(x, epsilon=1e-7, axis=(1,2)):
    x -= np.mean(x, axis=axis, keepdims=True)
    x /= np.std(x, axis=axis, keepdims=True) + epsilon

def create_generators(images, masks, data_dir, batch_size, train_indexes, val_indexes,                      validation_split=0.0, mask='inner',
                      shuffle_train_val=True, shuffle=True, seed=None,
                      normalize_images=True, augment_training=False,
                      augment_validation=False, augmentation_args={}):
    
    #images, masks = RVSC.load(data_dir, mask)
    
    for i in range(len(images)):
        # before: type(masks) = uint8 and type(images) = uint16
        # convert images to double-precision
        images[i] = images[i].astype('float64')

        # maybe normalize image
        if normalize_images:
            normalize(images[i], axis=(1,2))

    if seed is not None:
        np.random.seed(seed)
    
    if shuffle_train_val and len(train_indexes) == 0: #NO effect if cross validation is performed
        # shuffle images and masks in parallel
        rng_state = np.random.get_state()
        np.random.shuffle(images)
        np.random.set_state(rng_state)
        np.random.shuffle(masks)

    if len(train_indexes) != 0: #if cross validation is performed
        images = np.array(images)
        masks = np.array(masks)
        if augment_training:
            train_generator = Iterator(
                images[train_indexes], masks[train_indexes], batch_size, 'train', shuffle=shuffle, **augmentation_args)
            
        else:
            train_generator = Generator(images[train_indexes], masks[train_indexes], batch_size, 'train', shuffle = shuffle)

        train_steps_per_epoch = ceil(len(train_indexes) / batch_size)

        if augment_validation:
            val_generator = Iterator(
                images[val_indexes], masks[val_indexes], batch_size, 'val', shuffle=shuffle, **augmentation_args)
        else:
            val_generator = Generator(images[val_indexes], masks[val_indexes], batch_size, 'val', shuffle = shuffle)

        val_steps_per_epoch = ceil(len(val_indexes) / batch_size)

    else:
        # split out last %(validation_split) of images as validation set
        split_index = int((1 - validation_split) * len(images))

        print("training set size:", len(images[:split_index]))
        print("validation set size:", len(images[split_index:]))

        if augment_training:
            train_generator = Iterator(
                images[:split_index], masks[:split_index], batch_size, 'train', shuffle=shuffle, **augmentation_args)
            
        else:
            train_generator = Generator(images[:split_index], masks[:split_index], batch_size, 'train', shuffle = shuffle)

        train_steps_per_epoch = ceil(split_index / batch_size)

        if validation_split > 0.0:
            if augment_validation:
                val_generator = Iterator(
                    images[split_index:], masks[split_index:], batch_size, 'val', shuffle=shuffle, **augmentation_args)
            else:
                val_generator = Generator(images[split_index:], masks[split_index:],batch_size, 'val', shuffle = shuffle)
        else:
            val_generator = None

        val_steps_per_epoch = ceil((len(images) - split_index) / batch_size)

    return (train_generator, train_steps_per_epoch,
            val_generator, val_steps_per_epoch)
