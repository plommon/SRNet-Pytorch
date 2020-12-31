import cfg
from train.text_conversion import TextConversionTrainer

if __name__ == '__main__':
    TextConversionTrainer(cfg.data_backup_dir).train()
