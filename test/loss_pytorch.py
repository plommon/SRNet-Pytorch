import torch

import cfg


def build_discriminator_loss(x):
    x_true, x_pred = torch.chunk(x, 2)
    d_loss = -torch.mean(torch.log(torch.clamp(x_true, cfg.epsilon, 1.0))
                         + torch.log(torch.clamp(1.0 - x_pred, cfg.epsilon, 1.0)))
    return d_loss


def build_dice_loss(x_t, x_o):
    intersection = torch.sum(x_t * x_o, dim=[1, 2, 3])
    union = torch.sum(x_t, dim=[1, 2, 3]) + torch.sum(x_o, dim=[1, 2, 3])
    return 1. - torch.mean((2 * intersection + cfg.epsilon) / (union + cfg.epsilon), dim=0)


def build_l1_loss_with_mask(x_t, x_o, mask):
    mask_ratio = 1. - torch.sum(mask) / torch.tensor(torch.numel(mask), dtype=torch.float32)
    l1 = torch.abs(x_t - x_o)
    return mask_ratio * torch.mean(l1 * mask) + (1. - mask_ratio) * torch.mean(l1 * (1. - mask))


def build_l1_loss(x_t, x_o):
    return torch.mean(torch.abs(x_t - x_o))


def build_gan_loss(x):
    x_true, x_pred = torch.chunk(x, 2)
    return -torch.mean(torch.log(torch.clamp(x_pred, cfg.epsilon, 1.0)))


def build_perceptual_loss(x):
    l = []
    for f in x:
        l.append(build_l1_loss(f[0], f[1]))
    l = torch.stack(l, dim=0)
    l = torch.sum(l)
    return l


def build_gram_matrix(x):
    a, b, c, d = x.size()
    feature = x.view(a, d, b * c)
    feature_t = feature.transpose(1, 2)
    gram = torch.bmm(feature, feature_t) / (torch.tensor(b * c * d, dtype=torch.float32))
    return gram


def build_style_loss(x):
    l = []
    record = []
    for f in x:
        f_shape = torch.numel(f[0])
        f_norm = 1. / torch.tensor(f_shape, dtype=torch.float32)
        gram_true = build_gram_matrix(f[0])
        gram_pred = build_gram_matrix(f[1])
        record.append(torch.sum(gram_true))
        l.append(f_norm * build_l1_loss(gram_true, gram_pred))
    l = torch.stack(l, dim=0)
    l = torch.sum(l)
    return l, record


def build_vgg_loss(x):
    split = []
    for f in x:
        split.append(torch.chunk(f, 2))
    l_per = build_perceptual_loss(split)
    l_style, record = build_style_loss(split)
    return l_per, l_style, record


def build_generator_loss(out_g, out_d, labels, out_vgg):
    o_sk, o_st, o_b, o_t, mask_c = out_g
    o_db, o_dt = out_d
    t_sk, i_st, i_b, i_t = labels
    o_vgg = out_vgg

    l_c_sk = 1. * build_dice_loss(t_sk, o_sk)
    l_c_l1 = build_l1_loss_with_mask(i_st, o_st, mask_c)
    l_c = l_c_l1 + l_c_sk

    l_b_gan = build_gan_loss(o_db)
    l_b_l1 = cfg.lb_beta * build_l1_loss(i_b, o_b)
    l_b = l_b_gan + l_b_l1

    l_t_gan = build_gan_loss(o_dt)
    l_t_l1 = 10. * build_l1_loss(i_t, o_t)
    l_t_vgg_per, l_t_vgg_style, record = build_vgg_loss(o_vgg)
    l_t_vgg_per = 1. * l_t_vgg_per
    l_t_vgg_style = 500. * l_t_vgg_style
    l_t = l_t_gan + l_t_l1 + l_t_vgg_per + l_t_vgg_style

    l = 1. * l_c + 1. * l_b + 1. * l_t
    return l, [l_c_sk, l_c_l1, l_b_gan, l_b_l1, l_t_gan, l_t_l1, l_t_vgg_per, l_t_vgg_style, record]


if __name__ == '__main__':
    a = torch.full((1, 3, 4, 5), 0.012)
    b = torch.full((1, 3, 4, 5), 0.023)
    c = torch.full((1, 3, 4, 5), 0.034)
    d = torch.full((1, 3, 4, 5), 0.045)
    e = torch.full((1, 3, 4, 5), 0.056)
    o_g = [a, b, c, d, e]

    f = torch.full((2, 3, 4, 5), 0.001)
    g = torch.full((2, 3, 4, 5), 0.004)
    o_d = [f, g]

    h = torch.full((1, 3, 4, 5), 0.6)
    i = torch.full((1, 3, 4, 5), 0.07)
    j = torch.full((1, 3, 4, 5), 0.08)
    k = torch.full((1, 3, 4, 5), 0.007)
    la = [h, i, j, k]

    m = torch.cat([a, b], dim=0)
    n = torch.cat([a, c], dim=0)
    o = torch.cat([b, c], dim=0)
    p = torch.cat([c, d], dim=0)
    q = torch.cat([d, e], dim=0)
    ovgg = [m, n, o, p, q]
    print(build_generator_loss(o_g, o_d, la, ovgg))

    print(build_gram_matrix(g))
