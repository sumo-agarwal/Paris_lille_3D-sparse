# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import trimesh
from pathlib import Path
from os.path import splitext
import warnings
from abc import abstractmethod

from . import visualize
from .utils import log, warn_always
from .core import is_listy, Collection


class ItemBase():
    def __init__(self, data, *args, **kwargs):
        self.data = data
        self._affine_mat = None
        self.verbose = 0
        super().__init__(*args, **kwargs)

    def __repr__(self):
        return f'{self.__class__.__name__} {str(self)}'

    def apply_tfms(self, tfms: Collection, do_resolve: bool = True, verbose=0, refresh_always=False, **kwargs):
        "Apply data augmentation with `tfms` to this `ItemBase`."

        verbose_bak = self.verbose

        # disturb random state flow
        # if do_resolve:
        #    _resolve_tfms(tfms)

        # if flow of affine transforms is ended, then refresh (apply)
        is_affine = [getattr(tfm.tfm, '_wrap', None) == 'affine' for tfm in tfms]
        is_do_refresh = np.diff(is_affine, append=0) == -1

        # x = self.clone()
        x = self
        for tfm, do_refresh in zip(tfms, is_do_refresh):

            if do_resolve:
                tfm.resolve()

            x.verbose = verbose

            x = tfm(x)
            if refresh_always or do_refresh:
                x.refresh()

        self.verbose = verbose_bak
        return x

    @property
    def affine_mat(self):
        "Get the affine matrix that will be applied by `refresh`."
        if self._affine_mat is None:
            # Transformation matrix in homogeneous coordinates for 3D is 4x4.
            self._affine_mat = np.eye(4).astype(np.float32)
            self._mat_list = []
        return self._affine_mat

    @affine_mat.setter
    def affine_mat(self, v) -> None:
        self._affine_mat = v

    def affine(self, func, *args, **kwargs):
        "Equivalent to `self.affine_mat = self.affine_mat @ func()`."
        m = func(*args, **kwargs)
        assert m.shape == (4, 4)
        if self.verbose:
            print("* affine:", func.__name__)
            print("affine_mat: was:")
            print(repr(self.affine_mat))
            print("m:")
            print(repr(m))

        # fixed order
        # self.affine_mat = self.affine_mat @ m
        self.affine_mat = m @ self.affine_mat
        self._mat_list += [m]

        if self.verbose:
            print("affine_mat: became:")
            print(repr(self.affine_mat))
        return self

    def refresh(self):
        "Apply affine (and others) transformations that have been sent to and store in the `ItemBase`."
        if self._affine_mat is not None:
            if self.verbose:
                print('refresh:', self._affine_mat)
            self.aplly_affine(self._affine_mat)
            self.last_affine_mat = self._affine_mat
            self._affine_mat = None
        return self

    @abstractmethod
    def aplly_affine(self, affine_mat):
        "Apply affine (and others) transformations that have been sent to and store in the `ItemBase`."
        # https://en.wikipedia.org/wiki/Transformation_matrix#Affine_transformations
        # https://pages.mtu.edu/~shene/COURSES/cs3621/NOTES/geometry/geo-tran.html
        # http://qaru.site/questions/144684/difference-between-numpy-dot-and-python-35-matrix-multiplication

    @abstractmethod
    def show(self, **kwargs):
        pass


class PointsItem(ItemBase):
    def __str__(self):
        # return str(self.obj)
        _id = self.data['id']
        _size = self.data['points'].shape

        return f"('{_id}', n: {_size[0]})"

    def copy(self):
        d = self.data.copy()
        o = PointsItem(d)
        return o

    def describe(self):
        d = self.data
        cn = self.__class__.__name__
        _id = d['id']
        print(f"{cn} ({_id})")
        log('points', d['points'])
        for k in ['labels', 'colors', 'normals', 'features','intensity']:
            v = d.get(k, None)
            if v is not None:
                log(k, v)

    @property
    def colors(self):
        return self.data.get('colors', None)

    @colors.setter
    def colors(self, v):
        self.data['colors'] = v

    @property
    def labels(self):
        return self.data.get('labels', None)
    @property
    def intensity(self):
        return self.data.get('intensity', None)
    
    @labels.setter
    def labels(self, v):
        self.data['labels'] = v

    def show(self, labels=None, colors=None, with_normals=False, point_size_value=1., normals_size_value=1., **kwargs):
        """Show"""
        d = self.data
        if labels is None:
            labels = d.get('labels', None)
        points = d['points']
        points = np.array(points, dtype=np.float64)

        normals = None
        if with_normals:
            normals = d.get('normals', None)

        colors = d.get('colors', colors)
        return visualize.scatter(points,
                                 labels=labels, colors=colors, normals=normals,
                                 point_size_value=point_size_value,
                                 vector_size_value=normals_size_value,
                                 **kwargs)

    def aplly_affine(self, affine_mat):
        "Apply affine (and others) transformations that have been sent to and store in the `ItemBase`."
        # https://en.wikipedia.org/wiki/Transformation_matrix#Affine_transformations
        # https://pages.mtu.edu/~shene/COURSES/cs3621/NOTES/geometry/geo-tran.html
        # http://qaru.site/questions/144684/difference-between-numpy-dot-and-python-35-matrix-multiplication

        # 3x3 for rotation, reflection, scale, shear
        m = affine_mat[:3, :3]
        # column 3x1 for transpose  (shifting)
        v = affine_mat[:3, 3]

        d = self.data

        points = d['points']
        normals = d.get('normals', None)

        points = np.matmul(points, m.T)   # = (m @ points.T).T
        points += v

        # incorrect, correct only for rotation
        if normals is not None:
            warnings.warn(
                'Item has normals, but normals affine transformation is not full implemented (only rotation, flip and transpose)')
            normals = np.dot(normals, m)

        d['points'] = points

        if normals is not None:
            d['normals'] = normals

        # TODO:
        # in common case, the normals are not transforms similar like points and vectors
        # normals is valid for rotation and flippings, but not for (not simmetric) scaling
        # https://paroj.github.io/gltut/Illumination/Tut09%20Normal%20Transformation.html
        # https://www.scratchapixel.com/lessons/mathematics-physics-for-computer-graphics/geometry/transforming-normals
        # In fact, the solution to transforming normals, is not to multiply them by the same matrix used for transforming points and vectors,
        # but to multiply them by the transpose of the inverse of that matrix


class SparseItem(ItemBase):
    def __str__(self):
        # return str(self.obj)
        return f"('{self.data['id']}')"

    def describe(self):
        d = self.data

        print('id:', d['id'])
        coords = self.data['coords']
        log('coords', coords)
        log('features', d['features'])
        log('x', coords[:, 0])
        log('y', coords[:, 1])
        log('z', coords[:, 2])
        if 'labels' in d:
            log('labels', d['labels'])

        n_voxels = self.num_voxels()
        n_points = len(coords)
        # print('points:', n_points)
        print('voxels:', n_voxels)
        print('points / voxels:', n_points / n_voxels)

    @property
    def labels(self):
        return self.data.get('labels', None)

    @labels.setter
    def labels(self, v):
        self.data['labels'] = v

    def num_voxels(self):
        coords = self.data['coords']
        df = pd.DataFrame(coords, columns=['x', 'y', 'z'])
        n_voxels = len(df.groupby(['x', 'y', 'z']).count())
        return n_voxels

    def show(self, labels=None, point_size_value=1., **kwargs):
        # The same as PointsItem.show but points = d['coords']
        d = self.data
        if labels is None:
            labels = d['labels']

        points = d['coords']
        points = np.array(points, dtype=np.float64)

        return visualize.scatter(points, labels=labels, point_size_value=point_size_value, **kwargs)

    def apply_tfms(self, tfms: Collection, **kwargs):
        "Subclass this method if you want to apply data augmentation with `tfms` to this `SparseItem`."
        if tfms:
            raise NotImplementedError(f" Transformation for {self.__class__.__name__} is not implemented.")
        return self


def extract_data(b: Collection):
    "Recursively map lists of items in `b ` to their wrapped data."
    if is_listy(b):
        return [extract_data(o) for o in b]
    return b.data if isinstance(b, ItemBase) else b
