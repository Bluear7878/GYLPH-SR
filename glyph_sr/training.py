"""Train the text-control branch with the recorded four-partition data recipe."""

from pathlib import Path

from .config import TrainingConfig, parse_config, validate_runtime


def parse_args(argv=None):
    return parse_config(TrainingConfig, argv)


def validate_inputs(config):
    validate_runtime(config)
    for name in (
        "hq_metadata",
        "negative_metadata",
        "validation_hq_metadata",
        "validation_negative_metadata",
        "model_config",
    ):
        value = getattr(config, name)
        if not value or not Path(value).is_file():
            raise FileNotFoundError(f"{name} must name an existing file: {value}")
    if config.ts_checkpoint and not Path(config.ts_checkpoint).is_file():
        raise FileNotFoundError(config.ts_checkpoint)
    if min(config.batch_size, config.epochs, config.devices, config.learning_rate) <= 0:
        raise ValueError("batch size, epochs, devices, and learning rate must be positive")
    if config.max_steps == 0 or config.max_steps < -1:
        raise ValueError("max_steps must be -1 or positive")


def run(config):
    validate_inputs(config)
    import pytorch_lightning as pl
    from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
    from pytorch_lightning.loggers import TensorBoardLogger

    from .data import make_hqlq_dataloader
    from .models.controlnet import load_controlnet

    Path(config.log_dir).mkdir(parents=True, exist_ok=True)
    Path(config.checkpoint_dir).mkdir(parents=True, exist_ok=True)
    if config.seed >= 0:
        pl.seed_everything(config.seed, workers=True)
    train = make_hqlq_dataloader(
        hq_meta_path=config.hq_metadata,
        neg_meta_path=config.negative_metadata,
        batch_size=config.batch_size,
        img_size=config.train_image_size,
        include_types=("P_HQ", "P_LQ", "N_HQ", "N_LQ"),
        shuffle=True,
    )
    validation = make_hqlq_dataloader(
        hq_meta_path=config.validation_hq_metadata,
        neg_meta_path=config.validation_negative_metadata,
        batch_size=1,
        img_size=config.validation_image_size,
        include_types=("P_HQ",),
        shuffle=False,
    )
    model, _ = load_controlnet(
        config.model_config,
        config,
        device="cuda:0",
        sign=config.variant,
        pretrained=config.ts_checkpoint or None,
    )
    model.learning_rate = config.learning_rate
    checkpoint = ModelCheckpoint(
        dirpath=config.checkpoint_dir,
        filename=config.checkpoint_filename,
        monitor=config.monitor_metric,
        mode=config.monitor_mode,
        save_top_k=1,
        save_last=True,
        every_n_epochs=1,
        save_on_train_epoch_end=True,
        auto_insert_metric_name=False,
    )
    trainer = pl.Trainer(
        accelerator="gpu",
        devices=config.devices,
        precision=config.precision,
        max_epochs=config.epochs,
        max_steps=config.max_steps,
        gradient_clip_val=config.gradient_clip_val,
        log_every_n_steps=1,
        check_val_every_n_epoch=1,
        num_sanity_val_steps=0,
        logger=TensorBoardLogger(save_dir=config.log_dir, name="text_sr_controlnet"),
        callbacks=[checkpoint, LearningRateMonitor(logging_interval="step")],
    )
    trainer.fit(model, train, validation)
    return trainer


def main(argv=None):
    run(TrainingConfig(**vars(parse_args(argv))))
    return 0
