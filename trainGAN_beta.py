import argparse
import torch.nn as nn
from tensorboardX import SummaryWriter
from tqdm import tqdm
from util.evaluation import evaluate
from network.loss import InpaintingLoss
from network.net import PConvUNet, VGG16FeatureExtractor, NLayerDiscriminator
from util.io import load_ckpt
from util.io import save_ckpt
from util.data import *


parser = argparse.ArgumentParser()
# training options
parser.add_argument('--root', type=str, default='/mnt/storage/workspace/roberts/nuscene/dataset_inpaint')
parser.add_argument('--save_dir', type=str, default='/mnt/storage/workspace/roberts/inpainting/snapshots_GAN')
parser.add_argument('--log_dir', type=str, default='/mnt/storage/workspace/roberts/inpainting/logs_GAN')
parser.add_argument('--lr', type=float, default=2e-4)
parser.add_argument('--lr_finetune', type=float, default=5e-5)
parser.add_argument('--max_iter', type=int, default=1000000)
parser.add_argument('--batch_size', type=int, default=6)
parser.add_argument('--n_threads', type=int, default=5)
parser.add_argument('--save_model_interval', type=int, default=50000)
parser.add_argument('--vis_interval', type=int, default=1000)
parser.add_argument('--log_interval', type=int, default=100)
parser.add_argument('--resume', type=str)
parser.add_argument('--finetune', action='store_true')
parser.add_argument('--alpha', type=float, default=0.5)
parser.add_argument('--beta', type=float, default=1.0)
args = parser.parse_args()

torch.backends.cudnn.benchmark = True
device = torch.device('cuda')

if not os.path.exists(args.save_dir):
    # os.makedirs('{:s}/images_GAN_alpha_{:.2f}'.format(args.save_dir, args.alpha))
    # os.makedirs('{:s}/ckpt_GAN_alpha_{:.2f}'.format(args.save_dir, args.alpha))
    os.makedirs('{:s}/images_GAN_beta_{:.2f}'.format(args.save_dir, args.beta))
    os.makedirs('{:s}/ckpt_GAN_beta_{:.2f}'.format(args.save_dir, args.beta))


if not os.path.exists(args.log_dir):
    os.makedirs(args.log_dir)
writer = SummaryWriter(log_dir=args.log_dir)

# define datasets and dataloader
dataset_train = NusceneDataset(root_dir=args.root, train=True)
dataset_val = NusceneDataset(root_dir=args.root, train=False)

iterator_train = iter(data.DataLoader(
    dataset_train,
    batch_size=args.batch_size,
    sampler=InfiniteSampler(len(dataset_train)),
    num_workers=args.n_threads
))

# define generator and discriminator
modelG = PConvUNet(input_channels=3).to(device)
modelD = NLayerDiscriminator(6).to(device)

if args.finetune:
    lr = args.lr_finetune
    modelG.freeze_enc_bn = True
else:
    lr = args.lr

start_iter = 0

# define optimizers and criterion
optimizerG = torch.optim.Adam(filter(lambda p: p.requires_grad, modelG.parameters()), lr=lr, betas=(0.5, 0.999))
optimizerD = torch.optim.Adam(modelD.parameters(), lr=lr, betas=(0.5, 0.999))
criterionG = InpaintingLoss(VGG16FeatureExtractor()).to(device)
criterionD = nn.BCEWithLogitsLoss().to(device)

# resume from checkpoint
if args.resume:
    start_iter = load_ckpt(
        args.resume, [('model', modelG)],
        [('optimizer', optimizerG)]
    )
    for param_group in optimizerG.param_groups:
        param_group['lr'] = lr
    print('Starting from iter ', start_iter)



# training loop
for i in tqdm(range(start_iter, args.max_iter)):
    modelG.train()
    modelD.train()

    for k in range(1):
        sample = next(iterator_train)
        image, mask, gt = sample["img"], sample["mask"], sample["gt"]
        image = image.float().to(device)
        mask = mask.float().to(device)
        gt = gt.float().to(device)

        # train discriminator
        real_pair = torch.cat((image, gt), dim=1)

        with torch.no_grad():
            noise = torch.randn(image.shape[0], 1, image.shape[2], image.shape[3]).to(device)
            output, _ = modelG(image, mask)
            output = mask * image + (1 - mask) * output
            fake_pair = torch.cat((image, output), dim=1)

        # prediction
        real_pred = modelD(real_pair)
        fake_pred = modelD(fake_pair)

        # define labels
        real_labels = torch.ones_like(real_pred).to(device)
        fake_labels = torch.zeros_like(fake_pred).to(device)

        # optimize discriminator
        optimizerD.zero_grad()
        lossD_real, lossD_fake = criterionD(real_pred, real_labels), criterionD(fake_pred, fake_labels)
        lossD = (lossD_real + lossD_fake) / 2.0
        lossD.backward()
        optimizerD.step()

    sample = next(iterator_train)
    image, mask, gt = sample["img"], sample["mask"], sample["gt"]
    image = image.float().to(device)
    mask = mask.float().to(device)
    gt = gt.float().to(device)

    # train generator
    output, _ = modelG(image, mask)

    # compute and log losses
    loss_dict = criterionG(image, mask, output, gt)
    loss = 0.0

    for key, coef in opt.LAMBDA_DICT.items():
        value = coef * loss_dict[key]
        loss += value
        if (i + 1) % args.log_interval == 0:
            writer.add_scalar('loss_{:s}'.format(key), value.item(), i + 1)

    output = mask * gt + (1 - mask) * output
    fake_pair = torch.cat((image, output), dim=1)
    fake_pred = modelD(fake_pair)
    print(fake_pred.shape)

    # add loss from generator
    lossG = criterionD(fake_pred, real_labels)
    loss_total = args.beta * loss + lossG

    if (i + 1) % args.log_interval == 0:
        writer.add_scalar('loss_G', lossG.item(), i + 1)
        writer.add_scalar('loss_D_real', lossD_real.item(), i + 1)
        writer.add_scalar('loss_D_fake', lossD_fake.item(), i + 1)
        writer.add_scalar('loss_D', lossD.item(), i + 1)
        writer.add_scalar('loss_total', loss_total.item(), i + 1)
        writer.add_scalar('loss_rec', loss.item(), i + 1)

    # optimization step
    optimizerG.zero_grad()
    loss_total.backward()
    optimizerG.step()

    # checkpoints
    if (i + 1) % args.save_model_interval == 0 or (i + 1) == args.max_iter:
        save_ckpt('{:s}/ckpt_GAN_beta_{:.2f}/{:d}.pth'.format(args.save_dir, args.beta, i + 1),
                   [('model', modelG)], [('optimizer', optimizerG)], i + 1)

    # outputs snapshots
    if (i + 1) % args.vis_interval == 0:
        modelG.eval()
        evaluate(modelG, dataset_val, device,
                  '{:s}/images_GAN_beta_{:.2f}/test_{:d}.jpg'.format(args.save_dir, args.beta, i + 1))

writer.close()
