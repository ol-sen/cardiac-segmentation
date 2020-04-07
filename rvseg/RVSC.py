import os
import glob
import numpy as np

from keras import utils

from . import patient


def load(data_dir, mask = 'both'):
    """Load all patient images and contours from TrainingSet, Test1Set or
    Test2Set directory. The directories and images are read in sorted order.

    Arguments:
      data_dir - path to data directory (TrainingSet, Test1Set or Test2Set)

    Output:
      tuples of (images, masks), both of which are 4-d tensors of shape
      (batchsize, height, width, channels). Images is uint16 and masks are
      uint8 with values 0 or 1.
    """
    assert mask in ['inner', 'outer', 'both']

    glob_search = os.path.join(data_dir, "patient*")
    patient_dirs = sorted(glob.glob(glob_search))
    if len(patient_dirs) == 0:
        raise Exception("No patient directories found in {}".format(data_dir))

    # load images and masks into memory
    images = []
    masks = []
    for patient_dir in patient_dirs:
        p = patient.PatientData(patient_dir)
        images += [np.asarray(p.images)[:,:,:,None]]
        if mask == 'inner':
            masks += [np.asarray(p.endocardium_masks)]
        elif mask == 'outer':
            masks += [np.asarray(p.epicardium_masks)]
        elif mask == 'both':
            masks += [np.asarray(p.endocardium_masks) + np.asarray(p.epicardium_masks)]
 
    for i in range(len(images)): 
        # one-hot encode masks
        dims = masks[i].shape
        classes = len(set(masks[i][0].flatten())) # get num classes from first image
        new_shape = dims + (classes,)        
        masks[i] = utils.to_categorical(masks[i]).reshape(new_shape)

    return images, masks
