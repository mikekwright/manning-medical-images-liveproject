import numpy as np


class Random3dCrop:
    def __init__(self, crop_size, seed=42):
        if type(crop_size) == int:
            crop_size = (crop_size, crop_size)
        self.crop_size = crop_size
        self.random = np.random.RandomState(seed=seed)
        
    def calculate_image_offsets(self, image_shape):
        x_size = image_shape[0] - self.crop_size[0]
        x_offset = self.random.choice(x_size)
        
        y_size = image_shape[1] - self.crop_size[1]
        y_offset = self.random.choice(y_size)
        
        return (x_offset, x_offset+self.crop_size[0], y_offset, y_offset + self.crop_size[1])
        
    def __call__(self, data):
        src_image, target_image = data
        assert src_image.shape == target_image.shape
        if self.crop_size[0] > src_image.shape[0] or self.crop_size[1] > src_image.shape[1]:
            raise Exception("The crop size is to large for the supplied image")
        
        image_offsets = self.calculate_image_offsets(src_image.shape)
        
        modified_src = src_image[image_offsets[0]:image_offsets[1], image_offsets[2]:image_offsets[3], :]
        modified_target = target_image[image_offsets[0]:image_offsets[1], image_offsets[2]:image_offsets[3], :]
        
        return (modified_src, modified_target)
    
    
class Random2dCrop:
    def __init__(self, crop_size, seed=42):
        if type(crop_size) == int:
            crop_size = (crop_size, crop_size)
        self.crop_size = crop_size
        self.random = np.random.RandomState(seed=seed)
        
    def calculate_image_offsets(self, image_shape):
        x_size = image_shape[0] - self.crop_size[0]
        x_offset = self.random.choice(x_size)
        
        y_size = image_shape[1] - self.crop_size[1]
        y_offset = self.random.choice(y_size)
        
        return (x_offset, x_offset+self.crop_size[0], y_offset, y_offset + self.crop_size[1])
        
    def __call__(self, data):
        src_image, target_image = data
        assert src_image.shape == target_image.shape
        if self.crop_size[0] > src_image.shape[0] or self.crop_size[1] > src_image.shape[1]:
            raise Exception("The crop size is to large for the supplied image")
        
        image_offsets = self.calculate_image_offsets(src_image.shape)
        
        modified_src = src_image[image_offsets[0]:image_offsets[1], image_offsets[2]:image_offsets[3]]
        modified_target = target_image[image_offsets[0]:image_offsets[1], image_offsets[2]:image_offsets[3]]
        
        return (modified_src, modified_target)
