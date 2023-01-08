import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers

from materials.data_module import DataModule
from methods.nn_module import NNModule


def main():
    tb_logger = pl_loggers.TensorBoardLogger(save_dir="/root/tf-logs", name="FlowNetS", version="l1")
    trainer = pl.Trainer(
        logger=tb_logger,
        max_epochs = 1000,
        gpus = 1,
        num_nodes = 1,
        strategy = None,        
    )
    data_module = DataModule()
    nn_module = NNModule()
    trainer.fit(model=nn_module, datamodule=data_module)


if __name__ == "__main__":
    main()