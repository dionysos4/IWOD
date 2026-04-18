from iwod.dataset import kitti_multiview_detection as kmd
import torch
import torchvision
from iwod.utils.transforms import *
from torch.utils.data import DataLoader
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint, Callback
from lightning.pytorch import seed_everything
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.callbacks import EarlyStopping
import os
from iwod.model.lightning_module import LitPSDepth
from iwod.utils.helper import load_config
import yaml
import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", help="Path to the config file", default="/home/dennis/git_repos/IWOD/config/train_cam2_cam0.yaml")
    args = parser.parse_args()
    CONFIG_PATH = args.config_path

    print(CONFIG_PATH)

    cfg = load_config(CONFIG_PATH)

    seed_everything(1, workers=True)
    if os.path.exists(os.path.join(cfg["log_root_dir"], cfg["log_dir"], cfg["log_version"])):
        raise ValueError("Version already exists. Change version in config file")

    

    # set train transforms based on config file
    train_transforms_list = []
    if cfg["thermal_noise"]:
        train_transforms_list.append(torchvision.transforms.RandomApply([AddThermalNoise(cfg)], p=cfg["p_noise"]))
    if cfg["black_img"]:
        train_transforms_list.append(torchvision.transforms.RandomApply([ZeroImage()], p=cfg["p_zero_out"]))
    if cfg["freeze"]:
        train_transforms_list.append(torchvision.transforms.RandomApply([ImageFreeze(cfg)], p=cfg["p_random_img"]))
    if cfg["all_failures"]:
        train_transforms_list.append(torchvision.transforms.RandomApply([
                                        torchvision.transforms.RandomChoice([
                                            AddThermalNoise(cfg),
                                            ImageFreeze(cfg),
                                            ZeroImage()
                                        ])
                                    ], p=cfg["p_random_img"]))
    train_transforms_list.append(torchvision.transforms.RandomApply([HorizontalFlipUnrectWithoutCamFlip()], p=cfg["p_horizontal_flip"]))
    train_transforms_list.append(Normalize(mean_ref=cfg["mean_ref"],
                                            std_ref=cfg["std_ref"],
                                            means=cfg["mean_target"],
                                            stds=cfg["std_target"]))
    train_transforms_list.append(PadImages((cfg["pad_image_h"], cfg["pad_image_w"])))
    train_transforms_list.append(CropImages((cfg["crop_image_h"], cfg["crop_image_w"])))
    train_transforms_list.append(ToTensor())

    transform_train=torchvision.transforms.Compose(train_transforms_list)


    train_dataset = kmd.KittiMultiviewDataset(cfg["data_directory"],
                                                training="train",
                                                transform=transform_train,
                                                cfg=cfg,
                                                cameras=cfg["cameras"])
    

    transform_val = torchvision.transforms.Compose([Normalize(mean_ref=cfg["mean_ref"],
                                            std_ref=cfg["std_ref"],
                                            means=cfg["mean_target"],
                                            stds=cfg["std_target"]),
                                            PadImages((cfg["pad_image_h"], cfg["pad_image_w"])),
                                            CropImages((cfg["crop_image_h"], cfg["crop_image_w"])),
                                            ToTensor()])
    

    val_dataset = kmd.KittiMultiviewDataset(cfg["data_directory"],
                                                training="valid",
                                                transform=transform_val,
                                                cfg=cfg,
                                                cameras=cfg["cameras"])
    
    dataloader_train = DataLoader(train_dataset, batch_size=cfg["batch_size"], shuffle=True, num_workers=cfg["workers"])
    dataloader_val = DataLoader(val_dataset, batch_size=cfg["batch_size"], shuffle=False, num_workers=cfg["workers"])


    LOG_ROOT = cfg["log_root_dir"]
    LOG_DIR = cfg["log_dir"]
    VERSION = cfg["log_version"]

    # saves top-K checkpoints based on "val_loss" metric
    checkpoint_dir = os.path.join(LOG_ROOT, LOG_DIR, VERSION)
    checkpoint_callback_val = ModelCheckpoint(
                            save_top_k=2,
                            monitor="validation/loss",
                            mode="min",
                            dirpath=checkpoint_dir,
                            filename="kitti-stereo-{epoch:02d}-{val_loss:.2f}",
    )

    checkpoint_callback_train = ModelCheckpoint(
                            save_top_k=-1,
                            monitor="training/loss",
                            mode="min",
                            dirpath=checkpoint_dir,
                            filename="kitti-stereo-{epoch:02d}-{train_loss:.2f}",
    )

    early_stop_callback = EarlyStopping(monitor="training/cls_loss", min_delta=cfg["early_stop_min_delta"], patience=cfg["early_stop_patience"], verbose=False, mode=cfg["early_stop_mode"])
    logger = TensorBoardLogger(LOG_ROOT, name=LOG_DIR, version=VERSION)
    model = LitPSDepth(cfg)

    trainer = L.Trainer(callbacks=[checkpoint_callback_val, checkpoint_callback_train, early_stop_callback], num_sanity_val_steps=cfg["num_sanity_val_steps"], 
                      accelerator=cfg["accelorator"], devices=cfg["gpu_devices"], strategy=cfg["strategy"], num_nodes=cfg["num_nodes"], precision=cfg["precision"], log_every_n_steps=cfg["log_every_n_stepss"],
                      logger=logger, max_epochs=cfg["max_epochs"], gradient_clip_val=cfg["gradient_clip_value"], gradient_clip_algorithm=cfg["gradient_clip_algorithm"])
    trainer.fit(model, dataloader_train, dataloader_val)


if __name__ == "__main__":
    main()