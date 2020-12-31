import cfg
from train.fusion import FusionTrainer

if __name__ == '__main__':
    FusionTrainer(cfg.data_dir).train()
