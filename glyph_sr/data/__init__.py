"""Training datasets and metadata readers."""

__all__ = ["HQLQDataset", "make_hqlq_dataloader", "collate_training_samples"]


def __getattr__(name):
    if name in __all__:
        from . import training

        return getattr(training, name)
    raise AttributeError(name)
