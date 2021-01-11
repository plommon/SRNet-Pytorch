import os

import cv2
import numpy as np
import torch

import cfg
from datagen import srnet_datagen, get_input_data
from loss import build_l_t_loss, build_discriminator_loss
from model import TextConversionNet, DiscriminatorMixed
from utils import get_train_name, print_log, PrintColor, pre_process_img, save_result, get_log_writer

device = torch.device(cfg.gpu)


class TextConversionTrainer:
    def __init__(self, data_dir):
        self.data_iter = srnet_datagen(data_dir)

        self.G = TextConversionNet().to(device)
        self.g_optimizer = torch.optim.Adam(self.G.parameters(), lr=cfg.learning_rate)
        self.g_scheduler = torch.optim.lr_scheduler.ExponentialLR(self.g_optimizer,
                                                                  (cfg.decay_rate ** (1 / cfg.decay_steps)))
        self.D = DiscriminatorMixed(in_dim1=4, in_dim2=6).to(device)
        self.d_optimizer = torch.optim.Adam(self.D.parameters(), lr=cfg.learning_rate)
        self.d_scheduler = torch.optim.lr_scheduler.ExponentialLR(self.d_optimizer,
                                                                  (cfg.decay_rate ** (1 / cfg.decay_steps)))
        self.writer = None

    def train(self):
        train_name = 't_' + get_train_name()

        self.writer = get_log_writer(train_name)

        for step in range(cfg.max_iter):
            global_step = step + 1

            d_loss, g_loss, g_loss_detail, d_loss_detail = self.train_step(next(self.data_iter))

            # 打印loss信息
            if global_step % cfg.show_loss_interval == 0 or step == 0:
                print_log("step: {:>6d}   d_loss: {:>3.5f}   g_loss: {:>3.5f}".format(global_step, d_loss, g_loss))

            # 写tensorboard
            if global_step % cfg.write_log_interval == 0:
                self.write_summary(g_loss, g_loss_detail, d_loss, d_loss_detail, global_step)

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

        o_sk, o_t = self.G(i_t, i_s)

        i_dsk_true = torch.cat([t_sk, i_t], dim=1)
        i_dsk_pred = torch.cat([o_sk, i_t], dim=1)
        i_dsk = torch.cat([i_dsk_true, i_dsk_pred])

        i_dt_true = torch.cat([t_t, i_t], dim=1)
        i_dt_pred = torch.cat([o_t, i_t], dim=1)
        i_dt = torch.cat([i_dt_true, i_dt_pred], dim=0)

        # d_loss
        o_dsk, o_dt = self.D([i_dsk.detach(), i_dt.detach()])

        dt_loss = build_discriminator_loss(o_dt)
        dsk_loss = build_discriminator_loss(o_dsk)
        dt_loss_detail = [dt_loss, dsk_loss]
        d_loss = torch.add(dt_loss, dsk_loss)

        self.reset_grad()
        d_loss.backward()
        self.d_optimizer.step()

        # g_loss
        o_dsk, o_dt = self.D([i_dsk, i_dt])
        g_loss, g_loss_detail = build_l_t_loss(o_sk, o_t, o_dt, o_dsk, t_sk, t_t, mask_t)

        self.reset_grad()
        g_loss.backward()
        self.g_optimizer.step()

        self.d_scheduler.step()
        self.g_scheduler.step()

        return d_loss, g_loss, g_loss_detail, dt_loss_detail

    def reset_grad(self):
        self.d_optimizer.zero_grad()
        self.g_optimizer.zero_grad()

    def write_summary(self, g_loss, g_loss_detail, d_loss, d_loss_detail, step):
        self.writer.add_scalar('g_loss', g_loss, step)
        self.writer.add_scalar('l_t_gan', g_loss_detail[0], step)
        self.writer.add_scalar('l_sk_gan', g_loss_detail[1], step)
        self.writer.add_scalar('l_t_sk', g_loss_detail[2], step)
        self.writer.add_scalar('l_t_l1', g_loss_detail[3], step)

        self.writer.add_scalar('d_loss', d_loss, step)
        self.writer.add_scalar('dt_loss', d_loss_detail[0], step)
        self.writer.add_scalar('dsk_loss', d_loss_detail[1], step)

    def save_checkpoint(self, save_dir):
        os.makedirs(save_dir)
        torch.save(self.G.state_dict(), os.path.join(save_dir, 'TextConversion.pth'))

    def predict(self, i_t, i_s, to_shape=None):
        assert i_t.shape == i_s.shape and i_t.dtype == i_s.dtype

        i_t, i_s, to_shape = pre_process_img(i_t, i_s, to_shape)
        o_sk, o_t = self.G(i_t.to(device), i_s.to(device))

        o_sk = o_sk.data.cpu()
        o_t = o_t.data.cpu()

        transpose_vector = [0, 2, 3, 1]
        o_sk = o_sk.permute(transpose_vector).numpy()
        o_t = o_t.permute(transpose_vector).numpy()

        o_sk = cv2.resize((o_sk[0] * 255.).astype(np.uint8), to_shape, interpolation=cv2.INTER_NEAREST)
        o_t = cv2.resize(((o_t[0] + 1.) * 127.5).astype(np.uint8), to_shape)

        return o_sk, o_t

    def predict_data_list(self, save_dir, input_data_list, mode=2):
        for data in input_data_list:
            i_t, i_s, original_shape, data_name = data
            result = self.predict(i_t, i_s, original_shape)
            save_result(save_dir, result, data_name, mode=mode)
