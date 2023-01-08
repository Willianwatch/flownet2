import pytorch_lightning as pl
from torch.optim import Adam
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

from.networks.FlowNetS import FlowNetS
from .losses import L1Loss


class NNModule(pl.LightningModule):
    def __init__(self, ) -> None:
        super().__init__()

        self.model = FlowNetS(args=None, input_channels=6)
        self.loss = L1Loss(args=None)
        self.rgb_max = 255.0

    def training_step(self, batch, batch_idx: int):
        tb_logger = self.logger.experiment
        tb_logger : SummaryWriter

        inputs, target = batch
        rgb_mean = inputs.contiguous().view(inputs.size()[:2]+(-1,)).mean(dim=-1).view(inputs.size()[:2] + (1,1,1,))
        x = (inputs - rgb_mean) / self.rgb_max
        x = torch.cat( (x[:,:,0,:,:], x[:,:,1,:,:]), dim = 1)

        output = self.model(x)[0]
        output = F.upsample(output, size=target.shape[-2:], mode="bilinear", align_corners=True)

        loss, epe = self.loss(output, target)

        tb_logger.add_scalars(main_tag="train", tag_scalar_dict={"loss" : loss.item(), "epe" : epe.item()}, global_step=self.global_step)

        return {"loss" : loss}

    def configure_optimizers(self):
        return Adam(self.model.parameters(), lr=1e-4)
