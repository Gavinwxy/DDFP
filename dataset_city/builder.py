import logging

from .cityscapes import build_city_semi_loader, build_cityloader

logger = logging.getLogger("global")


def get_loader(cfg, seed=0):
    train_loader_sup, train_loader_unsup = build_city_semi_loader(cfg, seed=seed)
    val_loader = build_cityloader("val", cfg)
    logger.info("Get loader Done...")
    return train_loader_sup, train_loader_unsup, val_loader



