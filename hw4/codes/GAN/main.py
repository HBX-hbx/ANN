import GAN
from trainer import Trainer
from dataset import Dataset
from tensorboardX import SummaryWriter

from pytorch_fid import fid_score

from torchvision.utils import make_grid
from torchvision.utils import save_image

import torch
import torch.optim as optim
import os
import argparse

def polate(le, ri, polate_cnt, pair_cnt):
    # interpolate j from [le, ri]
    # polate_cnt: 10, the number of sample points for each pair
    # pari_cnt: the number of pairs
    # calc z = j * z1 + (1 - j) * z2
    imgs = None
    if not os.path.exists('polate_imgs'):
        os.mkdir('polate_imgs')
    for _ in range(pair_cnt):
        path = 'polate_le-%d_ri-%d.png' % (le, ri)
        z1 = torch.randn(1, args.latent_dim, 1, 1, device=device)
        z2 = torch.randn(1, args.latent_dim, 1, 1, device=device)
        fixed_noise = z2 # (10, 16, 1, 1)
        for i in range(1, polate_cnt):
            j = le + i * (ri - le) / (polate_cnt - 1)
            polate_z = j * z1 + (1 - j) * z2
            fixed_noise = torch.cat((fixed_noise, polate_z), 0)
        
        if imgs == None:
            imgs = netG(fixed_noise) # (10, 1, 32, 32)
        else:
            imgs = torch.cat((imgs, netG(fixed_noise)), 0)
    print(imgs.shape)
    imgs = make_grid(imgs, nrow=polate_cnt) * 0.5 + 0.5
    print('saving to %s' % os.path.join('polate_imgs', path))
    save_image(imgs, os.path.join('polate_imgs', path))


def generate_sample():
    fixed_noise = torch.randn(100, args.latent_dim, 1, 1, device=device)
    imgs = make_grid(netG(fixed_noise), nrow=10) * 0.5 + 0.5
    save_image(imgs, "samples.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--do_train', action='store_true')
    parser.add_argument('--no_cuda', action='store_true')
    parser.add_argument('--latent_dim', default=16, type=int)
    parser.add_argument('--generator_hidden_dim', default=16, type=int)
    parser.add_argument('--discriminator_hidden_dim', default=16, type=int)
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--num_training_steps', default=5000, type=int)
    parser.add_argument('--logging_steps', type=int, default=10)
    parser.add_argument('--saving_steps', type=int, default=1000)
    parser.add_argument('--learning_rate', default=0.0002, type=float)
    parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
    parser.add_argument('--data_dir', default='../data', type=str, help='The path of the data directory')
    parser.add_argument('--ckpt_dir', default='results', type=str, help='The path of the checkpoint directory')
    parser.add_argument('--log_dir', default='./runs', type=str)
    args = parser.parse_args()

    config = 'z-{}_h-{}_batch-{}_num-train-steps-{}'.format(args.latent_dim, args.generator_hidden_dim, args.batch_size, args.num_training_steps)
    args.ckpt_dir = os.path.join(args.ckpt_dir, config)
    args.log_dir = os.path.join(args.log_dir, config)
    device = torch.device('cuda' if torch.cuda.is_available() and not args.no_cuda else 'cpu')

    dataset = Dataset(args.batch_size, args.data_dir)
    netG = GAN.get_generator(1, args.latent_dim, args.generator_hidden_dim, device)
    netD = GAN.get_discriminator(1, args.discriminator_hidden_dim, device)
    tb_writer = SummaryWriter(args.log_dir)

    if args.do_train:
        optimG = optim.Adam(netG.parameters(), lr=args.learning_rate, betas=(args.beta1, 0.999))
        optimD = optim.Adam(netD.parameters(), lr=args.learning_rate, betas=(args.beta1, 0.999))
        trainer = Trainer(device, netG, netD, optimG, optimD, dataset, args.ckpt_dir, tb_writer)
        trainer.train(args.num_training_steps, args.logging_steps, args.saving_steps)

    restore_ckpt_path = os.path.join(args.ckpt_dir, str(max(int(step) for step in os.listdir(args.ckpt_dir))))
    netG.restore(restore_ckpt_path)
    
    # polate(0, 1, 10, 10)
    # generate_sample()

    num_samples = 3000
    real_imgs = None
    real_dl = iter(dataset.training_loader)
    while real_imgs is None or real_imgs.size(0) < num_samples:
        imgs = next(real_dl)
        if real_imgs is None:
            real_imgs = imgs[0]
        else:
            real_imgs = torch.cat((real_imgs, imgs[0]), 0)
    real_imgs = real_imgs[:num_samples].expand(-1, 3, -1, -1) * 0.5 + 0.5

    with torch.no_grad():
        samples = None
        while samples is None or samples.size(0) < num_samples:
            imgs = netG.forward(torch.randn(args.batch_size, netG.latent_dim, 1, 1, device=device))
            if samples is None:
                samples = imgs
            else:
                samples = torch.cat((samples, imgs), 0)
    samples = samples[:num_samples].expand(-1, 3, -1, -1) * 0.5 + 0.5
    samples = samples.cpu()

    fid = fid_score.calculate_fid_given_images(real_imgs, samples, args.batch_size, device)
    tb_writer.add_scalar('fid', fid)
    print("FID score: {:.3f}".format(fid), flush=True)