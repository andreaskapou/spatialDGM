import numpy as np
import torch
import pytorch_lightning as pl
import os
import os.path as osp
import random
import codecs
from typing import Any, Callable, Dict, List, Optional, Tuple
from PIL import Image
from urllib.error import URLError
from torchvision.datasets.utils import download_and_extract_archive, check_integrity
from torch.utils.data import random_split, DataLoader
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from torchvision.datasets import VisionDataset

class MnistDataModule(pl.LightningDataModule):
	"""
	MNIST DataModule implementation enabling rotation (or translation). 
	Check __get_item function for implementation details
    """

	def __init__(self, 
		data_dir: str = osp.join('..', '..', 'data'),
		dataset_name: str = "MNIST",
		batch_size: int = 32,
		modify: int = 1, # 0: none, 1: rotate
		num_workers: int = 0
	):
		super().__init__()
		self.data_dir = data_dir
		self.dataset_name = dataset_name
		self.batch_size = batch_size
		self.modify = modify
		self.num_workers = num_workers
		self.data_transform = transforms.Compose([transforms.ToTensor()])

		# self.dims is returned when you call dm.size()
        # Setting default dims here because we know them.
        # Could optionally be assigned dynamically in dm.setup()
		self.dims = (1, 28, 28)

	def prepare_data(self):
		# download
		MnistRotate(root=self.data_dir, dataset_name=self.dataset_name, train=True, 
			download=True, transform=self.data_transform)
		MnistRotate(root=self.data_dir, dataset_name=self.dataset_name, train=False, 
			download=True, transform=self.data_transform)

	def setup(self, stage: Optional[str]=None):
		# Assign Train/val split(s) for use in Dataloaders
		if stage in (None, 'fit'):
			self.train_set = MnistRotate(self.data_dir, dataset_name=self.dataset_name, 
				train=True, download=False, modify=self.modify, transform=self.data_transform)
			self.test_set = MnistRotate(self.data_dir, dataset_name=self.dataset_name, 
				train=False, download=False, modify=self.modify, transform=self.data_transform)
			# Dimensionality of each data point
			#self.dims = tuple(self.train[0][0].shape)

	def train_dataloader(self):
		return DataLoader(self.train_set, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)

	def val_dataloader(self):
		return DataLoader(self.test_set, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)


class MnistRotate(VisionDataset):
	"""`MNIST <http://yann.lecun.com/exdb/mnist/>`_ Dataset.
    Args:
        root (string): Root directory of dataset where ``dataset_name/processed/training.pt``
            and  ``dataset_name/processed/test.pt`` exist.
        dataset_name (string): Name of the dataset, default = MNIST
        train (bool, optional): If True, creates dataset from ``training.pt``,
            otherwise from ``test.pt``.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
    """

	mirrors = [
    	'http://yann.lecun.com/exdb/mnist/',
    	'https://ossci-datasets.s3.amazonaws.com/mnist/',
    ]

	resources = [
    	("train-images-idx3-ubyte.gz", "f68b3c2dcbeaaa9fbdd348bbdeb94873"),
    	("train-labels-idx1-ubyte.gz", "d53e105ee54ea40749a09fcbcd1e9432"),
    	("t10k-images-idx3-ubyte.gz", "9fb629c4189551a2d022fa330f9573f3"),
    	("t10k-labels-idx1-ubyte.gz", "ec29112dd5afa0611ce80d1b7f02629c")
    ]

	training_file = 'training.pt'
	test_file = 'test.pt'
	classes = ['0 - zero', '1 - one', '2 - two', '3 - three', '4 - four',
               '5 - five', '6 - six', '7 - seven', '8 - eight', '9 - nine']

	def __init__(self,
		root: str,
		dataset_name: str = "MNIST",
		train: bool = True,
		modify: int = 1,
		transform: Optional[Callable] = None,
		target_transform: Optional[Callable] = None,
		download: bool = False,
	) -> None:
		super(MnistRotate, self).__init__(root, transform=transform, target_transform=target_transform)
		self.dataset_name = dataset_name
		self.modify = modify
		#self.rot_deg = 0
		self.rotations_legit = torch.from_numpy(np.linspace(-np.pi / 2, np.pi / 2, 1000)).float()

		torch._C._log_api_usage_once(f"torchvision.datasets.{self.dataset_name}")
		self.train = train  # training set or test set

		if self._check_legacy_exist():
			self.data, self.targets = self._load_legacy_data()
			return

		if download:
			self.download()

		if not self._check_exists():
			raise RuntimeError('Dataset not found.' + ' You can use download=True to download it')

		self.data, self.targets = self._load_data()

	def _check_legacy_exist(self):
		processed_folder_exists = os.path.exists(self.processed_folder)
		if not processed_folder_exists:
			return False

		return all(
			check_integrity(os.path.join(self.processed_folder, file)) for file in (self.training_file, self.test_file)
		)

	def _load_legacy_data(self):
		# This is for BC only. We no longer cache the data in a custom binary, but simply read from the raw data
		# directly.
		data_file = self.training_file if self.train else self.test_file
		return torch.load(os.path.join(self.processed_folder, data_file))

	def _load_data(self):
		image_file = f"{'train' if self.train else 't10k'}-images-idx3-ubyte"
		data = read_image_file(os.path.join(self.raw_folder, image_file))

		label_file = f"{'train' if self.train else 't10k'}-labels-idx1-ubyte"
		targets = read_label_file(os.path.join(self.raw_folder, label_file))
		return data, targets

	def __getitem__(self, index):
		"""
		Args:
			index (int): Index
		Returns:
			img_rot: Rotated image
			img_class: Class label of specific image
			angle: Angle used to rotate image
		"""
		img = self.data[index]
		# For consistency with all other datasets to return a PIL Image
		img = Image.fromarray(img.numpy(), mode='L')

		# Apply modifications to the image
		if self.modify == 1:
			# Rotate
			if self.train is False:
				angle = self.rotations_legit[index % len(self.rotations_legit)]
			else:
				angle = random.choice(self.rotations_legit)
			angle_deg = angle * 180 / np.pi
			# Perform rotation
			img = TF.rotate(img=img, angle=angle_deg.item()) 
		elif self.modify > 1: 
			raise ValueError('Option {} is not implemented'.format(self.modify))
		else:
			angle = torch.tensor([0.0])

		# Apply transformation
		if self.transform is not None:
			img = self.transform(img)
		return img, self.targets[index], angle



	def __len__(self) -> int:
		return len(self.data)

	@property
	def raw_folder(self):
		return os.path.join(self.root, self.dataset_name, 'raw')

	@property
	def processed_folder(self):
		return os.path.join(self.root, self.dataset_name, 'processed')

	@property
	def class_to_idx(self) -> Dict[str, int]:
		return {_class: i for i, _class in enumerate(self.classes)}

	def _check_exists(self) -> bool:
		return all(
			check_integrity(os.path.join(self.raw_folder, os.path.splitext(os.path.basename(url))[0]))
			for url, _ in self.resources
		)

	def download(self) -> None:
		"""Download the MNIST data if it doesn't exist already."""

		if self._check_exists():
			return

		os.makedirs(self.raw_folder, exist_ok=True)

		# download files
		for filename, md5 in self.resources:
			for mirror in self.mirrors:
				url = "{}{}".format(mirror, filename)
				try:
					print("Downloading {}".format(url))
					download_and_extract_archive(
						url, download_root=self.raw_folder,
						filename=filename,
						md5=md5)
				except URLError as error:
					print("Failed to download (trying next):\n{}".format(error))
					continue
				finally:
					print()
				break
			else:
				raise RuntimeError("Error downloading {}".format(filename))

	def extra_repr(self) -> str:
		return "Split: {}".format("Train" if self.train is True else "Test")


def get_int(b: bytes) -> int:
    return int(codecs.encode(b, 'hex'), 16)


SN3_PASCALVINCENT_TYPEMAP = {
    8: (torch.uint8, np.uint8, np.uint8),
    9: (torch.int8, np.int8, np.int8),
    11: (torch.int16, np.dtype('>i2'), 'i2'),
    12: (torch.int32, np.dtype('>i4'), 'i4'),
    13: (torch.float32, np.dtype('>f4'), 'f4'),
    14: (torch.float64, np.dtype('>f8'), 'f8')
}


def read_sn3_pascalvincent_tensor(path: str, strict: bool = True) -> torch.Tensor:
    """Read a SN3 file in "Pascal Vincent" format (Lush file 'libidx/idx-io.lsh').
       Argument may be a filename, compressed filename, or file object.
    """
    # read
    with open(path, "rb") as f:
        data = f.read()
    # parse
    magic = get_int(data[0:4])
    nd = magic % 256
    ty = magic // 256
    assert 1 <= nd <= 3
    assert 8 <= ty <= 14
    m = SN3_PASCALVINCENT_TYPEMAP[ty]
    s = [get_int(data[4 * (i + 1): 4 * (i + 2)]) for i in range(nd)]
    parsed = np.frombuffer(data, dtype=m[1], offset=(4 * (nd + 1)))
    assert parsed.shape[0] == np.prod(s) or not strict
    return torch.from_numpy(parsed.astype(m[2], copy=False)).view(*s)

def read_label_file(path: str) -> torch.Tensor:
    x = read_sn3_pascalvincent_tensor(path, strict=False)
    assert(x.dtype == torch.uint8)
    assert(x.ndimension() == 1)
    return x.long()


def read_image_file(path: str) -> torch.Tensor:
    x = read_sn3_pascalvincent_tensor(path, strict=False)
    assert(x.dtype == torch.uint8)
    assert(x.ndimension() == 3)
    return x



