# -*- coding: utf-8 -*-

import numpy as np
import torch

from .main import Transform, transfer_keys
from ..data_items import PointsItem, SparseItem

from .. import utils

__all__ = ['TfmConvertItem','to_sparse_voxels', 'merge_features']


class TfmConvertItem(Transform):
    order = 0
    pass

def _to_sparse_voxels(x: PointsItem):
    d = x.data.copy()

    points = d['points']

    # TODO: is floor better then simply astype(np.int64) ? For x > 0 there is no differences
    # Some spreadsheet programs calculate the “floor-towards-zero”, in other words floor(-2.5) == -2. NumPy instead uses
    # the definition of floor where floor(-2.5) == -3.
    # >>> a = np.array([-1.7, -1.5, -0.2, 0.2, 0.5, 0.7, 1.3, 1.5, 1.7, 2.0, 2.5, 2.9])
    # >>> b = np.floor(a)
    # >>> c = a.astype(np.int64)
    # >>> pd.DataFrame([a, b, c])
    #    	-1.7 	-1.5 	-0.2 	0.2 	0.5 	0.7 	1.3 	1.5 	1.7 	2.0 	2.5 	2.9
    #    	-2.0 	-2.0 	-1.0 	0.0 	0.0 	0.0 	1.0 	1.0 	1.0 	2.0 	2.0 	2.0
    #    	-1.0 	-1.0 	0.0 	0.0 	0.0 	0.0 	1.0 	1.0 	1.0 	2.0 	2.0 	2.0

    coords = np.floor(points).astype(np.int64)

    # TODO: filter result, coords.min() must be >=0, warn

    labels = d['labels']
    is_multilabels = isinstance(labels, (list, tuple))
    if is_multilabels:
        labels_new = []
        for l in labels:
            labels_new.append(_convert_labels_dtype(l))
    else:
        labels = _convert_labels_dtype(labels)

    res = {'coords': coords,
           'features': d['features'],
           'labels': labels,
           }

    transfer_keys(d, res)

    return SparseItem(res)


def _convert_labels_dtype(x):
    if np.issubdtype(x.dtype, np.integer):
        return x.astype(np.int64)
    else:
        return x.astype(np.float32)


to_sparse_voxels = TfmConvertItem(_to_sparse_voxels)


def _merge_features(x: PointsItem, ones=False, intensity=False):
    # TODO: inplace

    append_ones = ones
    append_intensity = intensity
    d = x.data.copy()
    points = d['points']
    intensity = d['intensity']
    n_points = points.shape[0]
   
    # create features
    features = []
    if append_ones:
        
        npones=np.ones(n_points).astype(np.float32)
        features.append(npones)

    if append_intensity:
        if intensity is not None:
            features.append(intensity)
        else:
            utils.warn_always('merge_features: append_intensity is True, but there is no intensity')
    features = np.hstack(features)

    res = {'points': points, 'features': features, 'labels': d['labels']}

    # TODO: global/parameter ?
    transfer_keys(d, res)

    return PointsItem(res)


merge_features = TfmConvertItem(_merge_features)
