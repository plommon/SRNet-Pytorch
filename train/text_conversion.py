import os
import sys

import cv2
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

import cfg
from datagen import srnet_datagen, get_input_data
from loss import build_l_t_loss
from model import TextConversionNet
from utils import get_train_name, print_log, PrintColor, pre_process_img, save_result

device = torch.device(cfg.gpu)


class TextConversionTrainer:
    def __init__(self):
        self.data_iter = srnet_datagen()

        self.text_conversion_net = TextConversionNet().to(device)
        self.optimizer = torch.optim.Adam(self.text_conversion_net.parameters(), lr=cfg.learning_rate)
        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer,
                                                                (cfg.decay_rate ** (1 / cfg.decay_steps)))
        self.writer = None

    def train(self):
        train_name = 't_' + get_train_name()

        if sys.platform.startswith('win'):
            self.writer = SummaryWriter('model_logs\\train_logs\\' + train_name)
        else:
            self.writer = SummaryWriter(os.path.join(cfg.tensorboard_dir, train_name))

        for step in range(cfg.max_iter):
            global_step = step + 1

            loss, loss_detail = self.train_step(next(self.data_iter))

            # 打印loss信息
            if global_step % cfg.show_loss_interval == 0 or step == 0:
                print_log("step: {:>6d}   loss: {:>3.5f}".format(global_step, loss))

            # 写tensorboard
            if global_step % cfg.write_log_interval == 0:
                self.write_summary(loss, loss_detail, step)

            # 生成example
            if global_step % cfg.gen_example_interval == 0:
                save_dir = os.path.join(cfg.example_result_dir, train_name,
                                        'iter-' + str(global_step).zfill(len(str(cfg.max_iter))))
                self.predict_data_list(save_dir, get_input_data())
                print_log("example generated in dir {}".format(save_dir), content_color=PrintColor['green'])

            # 保存模型
            if global_step % cfg.save_ckpt_interval == 0:
                save_dir = os.path.join(cfg.checkpoint_save_dir, train_name,
                                        'iter-' + str(global_step).zfill(len(str(cfg.max_iter))))
                self.save_checkpoint(save_dir)
                print_log("checkpoint saved in dir {}".format(save_dir), content_color=PrintColor['green'])

    def train_step(self, data):
        i_t, i_s, t_sk, t_t, _, _, mask_t = data
        i_t = i_t.to(device)
        i_s = i_s.to(device)
        t_sk = t_sk.to(device)
        t_t = t_t.to(device)
        mask_t = mask_t.to(device)

        o_sk, o_t = self.text_conversion_net(i_t, i_s)

        loss, loss_detail = build_l_t_loss(o_sk, o_t, t_sk, t_t, mask_t)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.scheduler.step()

        return loss, loss_detail

    def write_summary(self, loss, loss_detail, step):
        self.writer.add_scalar('loss', loss, step)
        self.writer.add_scalar('l_t_sk', loss_detail[0], step)
        self.writer.add_scalar('l_t_l1', loss_detail[1], step)

    def save_checkpoint(self, save_dir):
        os.makedirs(save_dir)
        torch.save(self.text_conversion_net.state_dict(), os.path.join(save_dir, 'TextConversion.pth'))

    def predict(self, i_t, i_s, to_shape=None):
        assert i_t.shape == i_s.shape and i_t.dtype == i_s.dtype

        i_t, i_s, to_shape = pre_process_img(i_s, i_t, to_shape)
        o_sk, o_t = self.text_conversion_net(i_t.to(device), i_s.to(device))

        o_sk = o_sk.data.cpu()
        o_t = o_t.data.cpu()

        transpose_vector = [0, 2, 3, 1]
        o_sk = o_sk.permute(transpose_vector).numpy()
        o_t = o_t.permute(transpose_vector).numpy()

        o_sk = cv2.resize((o_sk[0] * 255.).astype(np.uint8), to_shape, interpolation=cv2.INTER_NEAREST)
        o_t = cv2.resize(((o_t[0] + 1.) * 127.5).astype(np.uint8), to_shape)

        return [o_sk, o_t]

    def predict_data_list(self, save_dir, input_data_list, mode=2):
        for data in input_data_list:
            i_t, i_s, original_shape, data_name = data
            result = self.predict(i_t, i_s, original_shape)
            save_result(save_dir, result, data_name, mode=mode)
