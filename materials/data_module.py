import pytorch_lightning as pl
from easydict import EasyDict
from torch.utils.data import DataLoader

from .datasets import FlyingChairs


class DataModule(pl.LightningDataModule):
    def __init__(self, ):
        super().__init__()
        args = EasyDict(
            crop_size = [256, 256],
            inference_size = [-1,-1],
        )
        self.args = args
        self.is_cropped = True
        self.root = "/root/autodl-tmp/data/FlyingChairs_release/data"

    def setup(self, stage = None) -> None:
        self.train_dataset = FlyingChairs(args=self.args, is_cropped=self.is_cropped, root=self.root)

    def train_dataloader(self):
        return DataLoader(dataset=self.train_dataset, 
                          batch_size=4, 
                          shuffle=True,
                          num_workers=4,
                          pin_memory=False,
                          drop_last=True)