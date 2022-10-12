import argparse
import json
import random
from time import time
import os
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from dataloader_example import create_data_loaders
from unet import create_model

import numpy as np
import torch
from torch import optim
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from test_radon_scikit import to_np
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
from skimage import color
import torch.optim.lr_scheduler as lr_scheduler



class Args(argparse.ArgumentParser):
    def __init__(self, ):
        super().__init__(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        # data param
        self.add_argument('--dataset_path', default='pd-unet-ct/big_data/not_circle/', type=str,
                          help='Path to the dataset raw files')

        # training param
        self.add_argument('--name', type=str, default='Erel_new-atoms',
                        help="Run name")
        self.add_argument('--restore', type=bool, default=None,
                help="If set will load the model according to name.")
        self.add_argument('--lr', type=float, default=1e-3,
                            help="Learning rate")
        self.add_argument('--tlr', type=float, default=1e-3,
                          help="Theta learning rate")
        self.add_argument('--gamma', type=float, default=0.5,
                          help="Theta optimizer gamma")
        self.add_argument('--num_epochs', type=int, default=100,
                            help="Number of epochs")
        self.add_argument('--transform', type=bool, default=False,
                          help='Adding augmentation during training.')
        self.add_argument('--normalize', type=bool, default=True,
                          help='Normalize the targets.')
        self.add_argument('--sample-rate', type=float, default=1.,
                          help='Fraction of total molecules to include')

        self.add_argument('--batch-size', type=int, default=10,
                          help='The size of the batch.')

        # Model parameters
        self.add_argument('--model', type=str, default='Unet',
                            help="String name of model")
        self.add_argument('--filter', type=str, default="ramp",
                          help="Name of iradon filter")
        self.add_argument('--num_pool_layers', type=int, default=4,
                            help="Number of poll layers - depth of the network")
        self.add_argument('--drop_prob', type=float, default=0.,
                            help="Dropout probability")
        self.add_argument('--num_channels', type=int, default=16,
                            help="Number of channels in middle layers")
        self.add_argument('--sparsity', type=int, default=16,
                          help="sparsity of the theta sampling")
        self.add_argument('--randt', type=bool, default=False,
                          help="random theta intitialization")

        self.add_argument('--num-workers', type=int, default=16,
                          help='Number of workers for each dataloader.')

        # Logging
        self.add_argument('--save_dir', type=str, default="summary/",
                help="Directory name to save models")


def psnr(gt, pred):
    """ Compute Peak Signal to Noise Ratio metric (PSNR) """
    gt = gt.detach().cpu().numpy()
    pred = pred.detach().cpu().numpy()
    return peak_signal_noise_ratio(gt, pred, data_range=gt.max())



def train_epoch(epoch, model, loss_fnc, dataloader, optimizer, args, writer, scheduler, optimizer_tlr):
    model.train()

    start_time = time()
    train_loss = []
    train_psnr = []
    with tqdm(dataloader, unit="batch") as tepoch:
        for i, x in enumerate(tepoch):
            x = x.to(args.device)
            g = model.sparse_backproject(model.sparse_transform(x))
            y = x
            # run model forward and compute loss
            pred = model(g)
            loss = loss_fnc(pred, y)
            psnr_val = psnr(y,pred)
            # backprop
            optimizer.zero_grad()
            optimizer_tlr.zero_grad()
            loss.backward()
            optimizer_tlr.step()
            optimizer.step()

            train_loss.append(loss.item())
            train_psnr.append(psnr_val.item())
            tepoch.set_postfix(loss=np.mean(train_loss).item())
        scheduler.step()


    t = to_np(torch.deg2rad(model.theta))
    fig = Figure()
    canvas = FigureCanvas(fig)
    ax = fig.gca()
    ax.scatter(np.sin(t), np.cos(t), cmap='gray')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect('equal')
    canvas.draw()
    #image = np.frombuffer(canvas.tostring_rgb(), dtype='uint8')
    #image = image.reshape(int(np.sqrt(len(image))), int(np.sqrt(len(image))))

    width, height = fig.get_size_inches() * fig.get_dpi()
    image = np.frombuffer(canvas.tostring_rgb(), dtype='uint8').reshape(int(height), int(width), 3)
    image = color.rgb2gray(image)

    print(f"[{epoch}|train] l2 loss: {np.mean(train_loss):.4f}+-{np.std(train_loss):.4f},"
                  f" in {int(time()-start_time)} secs")
    writer.add_scalar('Train Loss', np.mean(train_loss), epoch)
    writer.add_scalar('Train PSNR', np.mean(train_psnr), epoch)
    writer.add_image('Thetas', image, epoch, dataformats='HW')

def val_epoch(epoch, model, loss_fnc, dataloader, args, writer, scheduler_plat):
    model.eval()
    val_losses = []
    val_psnr = []
    pred_list = []
    gt_list = []
    with tqdm(dataloader, unit="batch") as tepoch:
        for i, x in enumerate(tepoch):
            x = x.to(args.device)

            # run model forward and compute loss
            with torch.no_grad():
                g = model.sparse_backproject(model.sparse_transform(x))
                y = x
                pred = model(g)
            loss = loss_fnc(pred, y)
            val_losses.append(loss.item())
            psnr_val=psnr(y,pred)
            val_psnr.append(psnr_val.item())
            pred_list.append(pred.cpu().numpy().squeeze())
            gt_list.append(y.cpu().numpy())
            tepoch.set_postfix(loss=np.mean(val_losses).item())
            #scheduler_plat.step(loss)

    print(f"...[{epoch}|val] l2 loss: {np.mean(val_losses):.4f}+-{np.std(val_losses):.4f}")
    writer.add_scalar('Val Loss', np.mean(val_losses), epoch)
    writer.add_scalar('Val PSNR', np.mean(val_psnr), epoch)
    if epoch % 5 == 0:
        samples = np.random.choice(len(gt_list), 2, replace=False)
        for i in samples:
            writer.add_image('Validation gt', gt_list[i][0], epoch, dataformats='CHW')
            writer.add_image('Validation pred', pred_list[i][0], epoch, dataformats='HW')
    return np.mean(val_losses)

def test_epoch(epoch, model, loss_fnc, dataloader, args, writer):
    model.eval()
    test_losses = []
    pred_list = []
    gt_list = []
    test_psnr = []
    with tqdm(dataloader, unit="batch") as tepoch:
        for i, x in enumerate(tepoch):
            x = x.to(args.device)

            # run model forward and compute loss
            with torch.no_grad():
                g = model.sparse_backproject(model.sparse_transform(x))
                y = x
                pred = model(g)
            loss = loss_fnc(pred, y)
            test_losses.append(loss.item())
            pred_list.append(pred.cpu().numpy().squeeze())
            gt_list.append(y.cpu().numpy())
            psnr_val=psnr(y,pred)
            test_psnr.append(psnr_val.item())
            tepoch.set_postfix(loss=np.mean(test_losses).item())

    print(f"...[{epoch}|test] l2 loss: {np.mean(test_losses):.4f}+-{np.std(test_losses):.4f}")

    #pred_list = np.concatenate(pred_list)
    #gt_list = np.concatenate(gt_list)
    #print(f'     , MSE: {np.abs(gt_list - pred_list).mean()}, '
    #          f'MSE: {((gt_list - pred_list) ** 2).mean()}')
    writer.add_scalar('Test Loss', np.mean(test_losses), epoch)
    writer.add_scalar('Test PSNR', np.mean(test_psnr), epoch)
    samples = np.random.choice(len(gt_list), 5, replace=False)
    for i in samples:
        writer.add_image('Test gt', gt_list[i][0], i, dataformats='CHW')
        writer.add_image('Test pred', pred_list[i][0], i, dataformats='HW')
    return np.mean(test_losses)

# Loss function
def task_loss(pred, target):
    # l1_loss = torch.mean(torch.abs(pred - target))
    l2_loss = torch.mean((pred - target)**2)
    return l2_loss

def main(args):
    # Prepare data
    train_loader, val_loader, test_loader = create_data_loaders(args)

    # Choose model
    model = create_model(args)

    # Optimizer settings
    optimizer = optim.Adam([{'params': model.parameters()}], lr=args.lr)
    optimizer_tlr = optim.Adam([{'params': model.theta}], lr=args.tlr)
    scheduler = lr_scheduler.MultiStepLR(optimizer_tlr, milestones=[20, 40, 60, 75, 90], gamma=args.gamma)
    scheduler_plat = lr_scheduler.ReduceLROnPlateau(optimizer_tlr, patience=2, factor=0.5)

    # Save path
    writer = SummaryWriter(log_dir=args.exp_dir)

    # Run training
    print('Begin training')
    best_val_loss = 1e9
    best_epoch = 0
    for epoch in range(args.num_epochs):
        train_epoch(epoch, model, task_loss, train_loader, optimizer, args, writer, scheduler, optimizer_tlr)
        val_loss = val_epoch(epoch, model, task_loss, val_loader, args, writer, scheduler_plat)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            torch.save(model.state_dict(), args.exp_dir + '/model.pt')

    print(f'{best_epoch=}, {best_val_loss=:.4f}')
    _ = test_epoch(0, model, task_loss, test_loader, args, writer)
    writer.close()


if __name__ == '__main__':
    torch.manual_seed(0)
    np.random.seed(0)
    random.seed(0)

    args = Args().parse_args()
    print(args.name)
    args.exp_dir = f'{args.save_dir}/{args.name}'
    # Create model directory
    if not os.path.isdir(args.exp_dir):
        os.makedirs(args.exp_dir)

    with open(args.exp_dir + '/args.txt', "w") as f:
        json.dump(args.__dict__, f, indent=2)

    # Automatically choose GPU if available
    args.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print("\n\nArgs:", args)
    main(args)
