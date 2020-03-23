import re
import os
import logging
import numpy as np
import nibabel as nib
from torch.utils.data import Dataset, DataLoader


class NifTi2dDataset(Dataset):
    """
    Usage: 
      src_path should be path to NifTi images
    """
    def __init__(self, src_path, target_path, layer_count=None, total_layers=None, transformer=None, seed=42):
        self.src_path = src_path
        self.target_path = target_path
        self.layer_count = layer_count
        self.total_layers = total_layers
        self.transformer = transformer or (lambda d: d)
        self.random = np.random.RandomState(seed=seed)
        self._items = None
    
    def __len__(self):
        return len(self.items)
    
    def __getitem__(self, key):
        result = self.load(self.items[key])
        
        return self.transformer(result)
    
    def load(self, item):
        src_file, target_file, layer = item
        logger.debug(f'Loading {src_file}, {target_file} - {layer}')
        src_img = self.load_image(src_file, self.src_path)
        target_img = self.load_image(target_file, self.target_path)
        return (src_img[:,:,layer], target_img[:,:,layer])
            
    @property
    def items(self):
        if self._items is None:
            logger.info('Loading item names from disk')
            src = self.list_items(self.src_path)
            target = self.list_items(self.target_path)
            
            items = [(s, t) for sname, s in src.items() for tname, t in target.items() if sname == tname]
            self._items = []
            for src, target in items:
                total_layers = self.total_layers or self.load_image(src, self.src_path).shape[2]
                if self.layer_count:
                    layer_num_list = list(range(total_layers))
                    self._items.extend([(src, target, i) for i in self.random.choice(layer_num_list, self.layer_count)])
                else:
                    self._items.extend([(src, target, i) for i in range(total_layers)])
                
            self.random.shuffle(self._items)
        
        return self._items
    
    def load_image(self, name, path):
        return nib.load(os.path.join(path, name)).get_fdata()

    def list_items(self, path):
        results = {}
        
        name_regex = r'^(?P<name>.*)\-T[1|2].*$'
        for filename in os.listdir(path):
            matches = re.match(name_regex, filename)
            if matches:
                results[matches['name']] = filename
        
        return results
    
    
class NifTiDataset(Dataset):
    """
    Usage: 
      src_path should be path to NifTi images
    """
    def __init__(self, src_path, target_path, transformer=None, seed=42):
        self.src_path = src_path
        self.target_path = target_path
        self.transformer = transformer or (lambda d: d)
        self.random = np.random.RandomState(seed=seed)
        self._items = None
    
    def __len__(self):
        return len(self.items)
    
    def __getitem__(self, key):
        result = self.load(self.items[key])
        
        return self.transformer(result)
    
    def load(self, item):
        src_file, target_file = item
        src_img = self.load_image(src_file, self.src_path)
        target_img = self.load_image(target_file, self.target_path)
        return (src_img, target_img)
            
    @property
    def items(self):
        if self._items is None:
            logger.info('Loading item names from disk')
            src = self.list_items(self.src_path)
            target = self.list_items(self.target_path)
            
            self._items = [(s, t) for sname, s in src.items() for tname, t in target.items() if sname == tname]               
            self.random.shuffle(self._items)
        
        return self._items
    
    def load_image(self, name, path):
        return nib.load(os.path.join(path, name)).get_fdata()

    def list_items(self, path):
        results = {}
        
        name_regex = r'^(?P<name>.*)\-T[1|2].*$'
        for filename in os.listdir(path):
            matches = re.match(name_regex, filename)
            if matches:
                results[matches['name']] = filename
        
        return results
    