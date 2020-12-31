import os

import cv2
import numpy as np
import torch

import cfg
from datagen import srnet_datagen, get_input_data
from loss import build_discriminator_loss, build_l_f_loss
from model import NewFusionNet, Discriminator, get_vgg_model
from utils import get_train_name, print_log, PrintColor, pre_process_img, save_result, get_log_writer

device = torch.device(cfg.gpu)


class FusionTrainer:
    def __init__(self, data_dir):
        self.data_iter = srnet_datagen(data_dir)

        self.vgg_selected_net = get_vgg_model().to(device)

        self.G = NewFusionNet().to(device)
        self.g_optimizer = torch.optim.Adam(self.G.parameters(), lr=cfg.learning_rate)
        self.g_scheduler = torch.optim.lr_scheduler.ExponentialLR(self.g_optimizer,
                                                                  (cfg.decay_rate ** (1 / cfg.decay_steps)))
        self.D = Discriminator().to(device)
        self.d_optimizer = torch.optim.Adam(self.D.parameters(), lr=cfg.learning_rate)
        self.d_scheduler = torch.optim.lr_scheduler.ExponentialLR(self.d_optimizer,
                                                                  (cfg.decay_rate ** (1 / cfg.decay_steps)))

        self.writer = None

    def train(self):
        train_name = 'f_' + get_train_name()

        self.writer = get_log_writer(train_name)

        for step in range(cfg.max_iter):
            global_step = step + 1

            d_loss, g_loss, g_loss_detail = self.train_step(next(self.data_iter))

            # 打印loss信息
            if global_step % cfg.show_loss_interval == 0 or step == 0:
                print_log("step: {:>6d}   d_loss: {:>3.5f}   g_loss: {:>3.5f}".format(global_step, d_loss, g_loss))

            # 写tensorboard
            if global_step % cfg.write_log_interval == 0:
                self.write_summary(g_loss, g_loss_detail, d_loss, global_step)

            # 生成example
            if global_step % cfg.gen_example_interval == 0:
                save_dir = os.path.join(cfg.example_result_dir, train_name,
                                        'iter-' + str(global_step).zfill(len(str(cfg.max_iter))))
                self.predict_data_list(save_dir, get_input_data(cfg.example_fusion_test_dir, is_fusion=True))
                print_log("example generated in dir {}".format(save_dir), content_color=PrintColor['green'])

            # 保存模型
            if global_step % cfg.save_ckpt_interval == 0:
                save_dir = os.path.join(cfg.checkpoint_save_dir, train_name,
                                        'iter-' + str(global_step).zfill(len(str(cfg.max_iter))))
                self.save_checkpoint(save_dir)
                print_log("checkpoint saved in dir {}".format(save_dir), content_color=PrintColor['green'])

    def train_step(self, data):
        i_t, _, _, t_t, t_b, t_f, _ = data

        i_t = i_t.to(device)
        t_t = t_t.to(device)
        t_b = t_b.to(device)
        t_f = t_f.to(device)

        o_f = self.G(t_t, t_b)

        i_df_true = torch.cat([t_f, i_t], dim=1)
        i_df_pred = torch.cat([o_f, i_t], dim=1)
        i_df = torch.cat([i_df_true, i_df_pred], dim=0)

        # d_loss
        # 在前面G的训练已经更新了其参数，这里不再更新G的参数，否则会导致变量inplace操作，使得反向传播出现错误
        o_df = self.D(i_df.detach())

        d_loss = build_discriminator_loss(o_df)

        self.reset_grad()
        d_loss.backward()
        self.d_optimizer.step()

        # g_loss
        o_df = self.D(i_df)

        i_vgg = torch.cat([t_f, o_f], dim=0)
        vgg_layers = [1, 6, 11, 20, 29]
        out_vgg = []
        X = i_vgg
        for i in range(len(self.vgg_selected_net)):
            X = self.vgg_selected_net[i](X)
            if i in vgg_layers:
                out_vgg.append(X)

        g_loss, g_loss_detail = build_l_f_loss(o_df, o_f, out_vgg, t_f)

        self.reset_grad()
        g_loss.backward()
        self.g_optimizer.step()

        self.g_scheduler.step()
        self.d_scheduler.step()

        return d_loss, g_loss, g_loss_detail

    def reset_grad(self):
        self.d_optimizer.zero_grad()
        self.g_optimizer.zero_grad()

    def write_summary(self, g_loss, g_loss_detail, d_loss, step):
        self.writer.add_scalar('g_loss', g_loss, step)
        self.writer.add_scalar('l_f_gan', g_loss_detail[0], step)
        self.writer.add_scalar('l_f_l1', g_loss_detail[1], step)
        self.writer.add_scalar('l_f_vgg_per', g_loss_detail[2], step)
        self.writer.add_scalar('l_f_vgg_style', g_loss_detail[3], step)

        self.writer.add_scalar('d_loss', d_loss, step)

    def save_checkpoint(self, save_dir):
        os.makedirs(save_dir)
        torch.save(self.G.state_dict(), os.path.join(save_dir, 'Fusion.pth'))

    def predict(self, t_t, t_b, to_shape=None):
        assert t_t.shape == t_b.shape and t_t.dtype == t_b.dtype

        t_t, t_b, to_shape = pre_process_img(t_b, t_t, to_shape)
        o_f = self.G(t_t.to(device), t_b.to(device))

        o_f = o_f.data.cpu()

        transpose_vector = [0, 2, 3, 1]
        o_f = o_f.permute(transpose_vector).numpy()

        o_f = cv2.resize(((o_f[0] + 1.) * 127.5).astype(np.uint8), to_shape)

        return o_f

    def predict_data_list(self, save_dir, input_data_list, mode=4):
        for data in input_data_list:
            t_t, t_b, original_shape, data_name = data
            result = self.predict(t_t, t_b, original_shape)
            save_result(save_dir, result, data_name, mode=mode)
