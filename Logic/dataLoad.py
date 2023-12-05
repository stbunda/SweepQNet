import torch
import torchvision
import torchvision.transforms as transforms
from torchvision.datasets.folder import has_file_allowed_extension
import os
from typing import Any, Callable, cast, Dict, List, Optional, Tuple, Union
from torch.utils.data import random_split
import numpy as np

def snp_file_loader(path):
        data = np.memmap(path, mode='r+').tobytes()
        ints_per_col = np.frombuffer(data, dtype=np.uint16, offset=0, count=1)[0]
        num_cols = np.frombuffer(data, dtype=np.int32, offset=2, count=1)[0]
        target_pos = np.frombuffer(data, dtype=np.float64, offset=6, count=1)[0]
        
        snp_data = np.unpackbits(np.frombuffer(data, dtype=np.uint8, offset=14, count=ints_per_col*num_cols))
        snp_matrix = snp_data.reshape((ints_per_col*8, num_cols), order='F')
        snp_tensor = torch.from_numpy(snp_matrix).float()
        
        bp_array = np.frombuffer(data, dtype=np.float32, offset=14+ints_per_col*num_cols, count=num_cols)
        bp_tensor = torch.from_numpy(bp_array.copy())
        bp_tensor = torch.broadcast_to(bp_tensor, snp_tensor.shape)
        
        sample = torch.stack([bp_tensor, bp_tensor, snp_tensor], dim=0).float()
        return sample, target_pos

class shuffleDim(torch.nn.Module):

    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, img):
        rand_indices = torch.randperm(img.shape[self.dim])
        img = torch.index_select(img, self.dim, rand_indices)
        return img

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(dim={self.dim})"
    
    
class CustomImageFolder(torchvision.datasets.DatasetFolder):
    
    def __init__(self, root, transform, mix_images):
        print(root)
        extensions = (".png",)
        loader = torchvision.datasets.folder.default_loader
            
        super().__init__(root, transform=transform, extensions=extensions, loader=loader)
        self.mix_images = mix_images
    
    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        
        path, target = self.samples[index]
        
        sample = self.loader(path)
        
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)
        
        # sample[0:2, :, :] = sample[0:2, :, :] / 100.0

        return path, sample, target
    
    
class NoClassImageFolder(CustomImageFolder):
    @staticmethod
    def find_classes(directory: str) -> Tuple[List[str], Dict[str, int]]:
        class_to_idx = {'generic': 0}
        return (0,), class_to_idx

    @staticmethod
    def make_dataset(
        directory: str,
        class_to_idx: Optional[Dict[str, int]] = None,
        extensions: Optional[Union[str, Tuple[str, ...]]] = None,
        is_valid_file: Optional[Callable[[str], bool]] = None,
    ) -> List[Tuple[str, int]]:
        """Generates a list of samples of a form (path_to_sample, class).

        See :class:`DatasetFolder` for details.

        Note: The class_to_idx parameter is here optional and will use the logic of the ``find_classes`` function
        by default.
        """
        directory = os.path.expanduser(directory)

        if class_to_idx is None:
            _, class_to_idx = NoClassImageFolder.find_classes(directory)
        elif not class_to_idx:
            raise ValueError("'class_to_index' must have at least one entry to collect any samples.")

        both_none = extensions is None and is_valid_file is None
        both_something = extensions is not None and is_valid_file is not None
        if both_none or both_something:
            raise ValueError("Both extensions and is_valid_file cannot be None or not None at the same time")

        if extensions is not None:

            def is_valid_file(x: str) -> bool:
                return has_file_allowed_extension(x, extensions)  # type: ignore[arg-type]

        is_valid_file = cast(Callable[[str], bool], is_valid_file)

        instances = []
        available_classes = set()
        for target_class in sorted(class_to_idx.keys()):
            class_index = class_to_idx[target_class]
            target_dir = directory
            if not os.path.isdir(target_dir):
                continue
            for root, _, fnames in sorted(os.walk(target_dir, followlinks=True)):
                for fname in sorted(fnames):
                    path = os.path.join(root, fname)
                    if is_valid_file(path):
                        item = path, class_index
                        instances.append(item)

                        if target_class not in available_classes:
                            available_classes.add(target_class)

        empty_classes = set(class_to_idx.keys()) - available_classes
        if empty_classes:
            msg = f"Found no valid file for the classes {', '.join(sorted(empty_classes))}. "
            if extensions is not None:
                msg += f"Supported extensions are: {extensions if isinstance(extensions, str) else ', '.join(extensions)}"
            raise FileNotFoundError(msg)

        return instances
    


def get_loader(data_path, batch_size, class_folders, shuffle, shuffle_row, mix_images, validation):
    
    transform_list = []
    transform_list.append(transforms.ToTensor())
    
    if shuffle_row:
        transform_list.append(shuffleDim(1))
    
    transform = transforms.Compose(transform_list)
    
    if not class_folders:
        dataset = NoClassImageFolder(root=data_path, transform=transform, mix_images=mix_images)
    else:
        dataset = CustomImageFolder(root=data_path, transform=transform, mix_images=mix_images)
        
    # added to filter training samples for detection training
    # if train_detect:
    #     idx = [i for i in range(len(dataset)) if 40 < int(os.path.split(dataset.samples[i][0])[1].split('_')[1]) < 60]
    #     dataset = torch.utils.data.Subset(dataset, idx)
        
    if validation:
        (dataset, val_set) = random_split(dataset, [0.85, 0.15]) # , torch.Generator().manual_seed(90)
        val_loader = torch.utils.data.DataLoader(val_set, batch_size=batch_size,
                                            shuffle=shuffle, num_workers=0)
        
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                            shuffle=shuffle, num_workers=0)
    
    if validation:
        return dataloader, val_loader
    else:
        return dataloader, None
