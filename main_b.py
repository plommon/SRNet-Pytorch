import cfg
from train.background_inpainting import BackgroundInpaintingTrainer

if __name__ == '__main__':
    BackgroundInpaintingTrainer(cfg.data_backup_dir).train()
