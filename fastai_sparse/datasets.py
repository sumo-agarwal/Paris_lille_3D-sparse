# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import numbers
import glob
import os
from pathlib import Path
from tqdm import tqdm
from dataclasses import dataclass
from plyfile import PlyData, PlyElement
# from abc import abstractmethod

from torch.utils.data import Dataset

from . import utils
from .core import defaults, Any, Collection, Callable, Optional, try_int, PathOrStr, listify, PreProcessors, show_some
from .data_items import PointsItem


@dataclass
class DataSourceConfig():
    root_dir: PathOrStr
    df: Any
    batch_size: int = 32
    num_workers: int = defaults.cpus
    init_numpy_random_seed: bool = True   # TODO: rename and comment ?

    file_ext: str = None
    ply_label_name: str = None
    ply_colors_from_vertices: bool = True
    ply_labels_from_vertices: bool = True
    ply_colors_from_vertices: bool = True
    def __post_init__(self):
        if not isinstance(self.root_dir, Path):
            self.root_dir = Path(self.root_dir)

        self.df = self.df.reset_index(drop=True)

        self._equal_keys = ['ply_colors_from_vertices',
                            'ply_labels_from_vertices']

        self._repr_keys = ['root_dir', 'batch_size', 'num_workers',
                           'file_ext', 'ply_label_name',
                           'init_numpy_random_seed']

        self.check()

    def check(self):
        assert self.root_dir.exists()

    def __repr__(self) -> str:
        lines = [f"{self.__class__.__name__};"]
        for key in self._repr_keys:
            value = getattr(self, key)
            if value is not None:
                lines.append(f'   {key}: {value}')

        value = len(self.df)
        lines.append(f' Items count: {value}')
        s = '\n'.join(lines)
        return s


class BaseDataset(Dataset):

    def __init__(self, items, source_config=None, path: PathOrStr = '.', reader_fn: Callable = None, **kwargs):
        """
        Parameters
        ----------
        items: Collection
            Filenames of examples. For fastai compatibility. (TODO: reorganize it)
        reader_fn: Callable
            Function (self, i, row) that return instance of ItemBase or its subclasses.


        """
        # TODO: store `list(df.values)` in items.
        self.items = items
        self.source_config = source_config
        self.df = source_config.df
        self.path = Path(path)
        self.reader_fn = reader_fn
        self.tfms = None
        self.tfmargs = None

    # .. key methods ..
    def __len__(self):
        assert len(self.source_config.df) == len(self.items)
        return len(self.items)

    def get(self, i):
        row = self.df.iloc[i]

        if self.reader_fn is not None:
            return self.reader_fn(i, row, self=self)
        else:
            return self._reader_fn(i, row)

    def __getitem__(self, idxs: int) -> Any:
        """Return instance ItemBase or subclasses. With transformations."""
        idxs = try_int(idxs)
        if isinstance(idxs, numbers.Integral):
            item = self.get(idxs)
            if self.tfms or self.tfmargs:
                item = item.apply_tfms(self.tfms, **self.tfmargs)
            return item
        # else: return self.new(self.items[idxs], xtra=index_row(self.xtra, idxs))
        else:
            raise NotImplementedError()

    def __add__(self, other):
        # return ConcatDataset([self, other])
        raise NotImplementedError()

    def __repr__(self) -> str:
        items = [self[i] for i in range(min(5, len(self.items)))]
        return f'{self.__class__.__name__} ({len(self.items)} items)\n{show_some(items)}\nPath: {self.path}'

    def transform(self, tfms: Collection, **kwargs):
        "Set the `tfms` to be applied to the inputs and targets."
        self.tfms = tfms
        self.tfmargs = kwargs
        return self

    # .. other methods ..
    def get_example_id(self, i):
        row = self.source_config.df.iloc[i]
        return row.example_id

    @classmethod
    def from_source_config(cls, source_config, reader_fn=None):
        fnames = []
        df = source_config.df
        t = tqdm(df.iterrows(), total=len(df), desc='Load file names')
        try:
            for i, row in t:
                fnames.append(cls.get_filename_from_row(row, source_config))
        finally:
            t.clear()
            t.close()
        o = cls(fnames, source_config=source_config,
                reader_fn=reader_fn,
                path=source_config.root_dir)
        return o

    @classmethod
    def get_filename_from_row(cls, row, source_config):
        fname = source_config.root_dir
        
        if 'subdir' in row.keys():
            fname = fname#####################################################################33
        ext = source_config.file_ext
        assert (ext is not None) or 'ext' in row.keys(
        ), "Define file_ext in config or column 'ext' in DataFrame'"
        if ext is None:
            ext = row.ext
        return fname / (row.example_id + ext)

    def check(self):
        self.check_files_exists()

    def check_files_exists(self, max_num_examples: Optional[int] = None, desc='Check files exist'):
        total = len(self)
        if max_num_examples is not None:
            total = int(max_num_examples)

        t = tqdm(self.items, total=total, desc=desc)
        try:
            for i, item in enumerate(t):
                assert item.exists()
                if max_num_examples is not None:
                    if i >= total:
                        break
        except Exception as e:
            # t.clear()
            t.close()
            print(item)
            raise e


class PointsDataset(BaseDataset):

    def _reader_fn(self, i, row):
        """Returns instance of ItemBase by its index and supplimented dataframe's row

           Default reader.
           Used if self.reader_fn is None.
        """
        data = {}
        data['id'] = self.get_example_id(i)
        fn = self.items[i]#self.get_filename(i)
        cloud = PlyData.read(fn)
        n=cloud.elements[0].count
        array = np.zeros(5*n).reshape(n,5) # initializing a numpy array for ply format
        for i in range(n):
            for j in range(5):
                array[i,j]=cloud.elements[0].data[i][j]
   
        data['points'] = array[:, 0:3]
        data['intensity']=array[:,3]#scalar_reflectance==intensity
        #print(array[:,3])
        data['labels'] = array[:,4]#scalar_class==labels
        #print(PointsItem(data))
        return PointsItem(data)