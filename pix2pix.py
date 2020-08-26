import networks
import argparse
import utils
import torch
import itertools
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from datasets import PairedImage
import time
import numpy as np



# Get training options from the command line
def get_opt():
    parser = argparse.ArgumentParser()
    # Training parameters
    parser.add_argument('--epoch', type=int, default=0, help='starting epoch')
    parser.add_argument('--n_epochs', type = int, default = 100, help = 'number of epochs with initial learning rate')
    parser.add_argument('--n_epochs_decay', type = int, default = 0, help = 'number of epochs starting the decay of learning rate')
    parser.add_argument('--beta1', type = float, default = 0.5, help = 'momentum term of the Adam optimizer')
    parser.add_argument('--lr', type = float, default = 0.0002, help = 'initial learning rate')
    parser.add_argument('--batch_size', type = int, default = 8, help = 'batch size of training')
    parser.add_argument('--cuda', action='store_true', help='use GPU computation')
    parser.add_argument('--rootdir', type=str, default='lsm/', help='root directory of the dataset')
    parser.add_argument('--n_cpu', type=int, default=4, help='number of cpu threads to use during batch generation')
    parser.add_argument('--u_net', action='store_true', help='use U-net generator')
    parser.add_argument('--pretrained', action='store_true', help='load pretrained weights')

    # Model parameters
    parser.add_argument('--sizeh', type=int, default=512, help='size of the image')
    parser.add_argument('--sizew', type=int, default=640, help='size of the image')
    parser.add_argument('--input_nc', type = int, default = 1, help = 'number of input channels')
    parser.add_argument('--output_nc', type = int, default = 1, help = 'number of output channels')
    parser.add_argument('--ngf', type = int, default = 64, help = 'number of filters in the generator')
    parser.add_argument('--ndf', type = int, default = 64, help = 'number of filters in the discriminator')
    parser.add_argument('--dropout', type = bool, default = False, help = 'whether to use dropout')
    parser.add_argument('--n_res', type = int, default = 9, help = 'number of resNet blocks')
    parser.add_argument('--l1_loss', type = float, default=10, help = 'coefficient of l1 loss')

    opt = parser.parse_args()
    return opt


def main():
    # Get training options
    opt = get_opt()

    device = torch.device("cuda") if opt.cuda else torch.device("cpu")

    # Define the networks
    # netG_A: used to transfer image from domain A to domain B
    netG_A = networks.Generator(opt.input_nc, opt.output_nc, opt.ngf, opt.n_res, opt.dropout)
    if opt.u_net:
        netG_A = networks.U_net(opt.input_nc, opt.output_nc, opt.ngf)

    # netD_B: used to test whether an image is from domain A
    netD_B = networks.Discriminator(opt.input_nc + opt.output_nc, opt.ndf)

    # Initialize the networks
    if opt.cuda:
        netG_A.cuda()
        netD_B.cuda()
    utils.init_weight(netG_A)
    utils.init_weight(netD_B)

    if opt.pretrained:
        netG_A.load_state_dict(torch.load('pretrained/netG_A.pth'))
        netD_B.load_state_dict(torch.load('pretrained/netD_B.pth'))


    # Define the loss functions
    criterion_GAN = utils.GANLoss()
    if opt.cuda:
        criterion_GAN.cuda()

    criterion_l1 = torch.nn.L1Loss()

    # Define the optimizers
    optimizer_G = torch.optim.Adam(netG_A.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
    optimizer_D_B = torch.optim.Adam(netD_B.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))

    # Create learning rate schedulers
    lr_scheduler_G = torch.optim.lr_scheduler.LambdaLR(optimizer_G, lr_lambda = utils.Lambda_rule(opt.epoch, opt.n_epochs, opt.n_epochs_decay).step)
    lr_scheduler_D_B = torch.optim.lr_scheduler.LambdaLR(optimizer_D_B, lr_lambda = utils.Lambda_rule(opt.epoch, opt.n_epochs, opt.n_epochs_decay).step)


    # Define the transform, and load the data
    transform = transforms.Compose([transforms.Resize((opt.sizeh, opt.sizew)),
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,))])
    dataloader = DataLoader(PairedImage(opt.rootdir, transform = transform, mode = 'train'), batch_size=opt.batch_size, shuffle=True, num_workers=opt.n_cpu)

    # numpy arrays to store the loss of epoch
    loss_G_array = np.zeros(opt.n_epochs + opt.n_epochs_decay)
    loss_D_B_array = np.zeros(opt.n_epochs + opt.n_epochs_decay)

    # Training
    for epoch in range(opt.epoch, opt.n_epochs + opt.n_epochs_decay):
        start = time.strftime("%H:%M:%S")
        print("current epoch :", epoch, " start time :", start)
        # Empty list to store the loss of each mini-batch
        loss_G_list = []
        loss_D_B_list = []

        for i, batch in enumerate(dataloader):
            if i % 20 == 1:
                print("current step: ", i)
                current = time.strftime("%H:%M:%S")
                print("current time :", current)
                print("last loss G_A:", loss_G_list[-1],  "last loss D_B:", loss_D_B_list[-1])

            real_A = batch['A'].to(device)
            real_B = batch['B'].to(device)

            # Train the generator
            utils.set_requires_grad([netG_A], True)
            optimizer_G.zero_grad()

            # Compute fake images and reconstructed images
            fake_B = netG_A(real_A)


            # discriminators require no gradients when optimizing generators
            utils.set_requires_grad([netD_B], False)


            # GAN loss
            prediction_fake_B = netD_B(torch.cat((fake_B, real_A), dim=1))
            loss_gan = criterion_GAN(prediction_fake_B, True)

            #L1 loss
            loss_l1 = criterion_l1(real_B, fake_B) * opt.l1_loss

            # total loss without the identity loss
            loss_G = loss_gan + loss_l1

            loss_G_list.append(loss_G.item())
            loss_G.backward()
            optimizer_G.step()

            # Train the discriminator
            utils.set_requires_grad([netG_A], False)
            utils.set_requires_grad([netD_B], True)

            # Train the discriminator D_B
            optimizer_D_B.zero_grad()
            # real images
            pred_real = netD_B(torch.cat((real_B, real_A), dim=1))
            loss_D_real = criterion_GAN(pred_real, True)

            # fake images
            fake_B = netG_A(real_A)
            pred_fake = netD_B(torch.cat((fake_B, real_A), dim=1))
            loss_D_fake = criterion_GAN(pred_fake, False)

            # total loss
            loss_D_B = (loss_D_real + loss_D_fake) * 0.5
            loss_D_B_list.append(loss_D_B.item())
            loss_D_B.backward()
            optimizer_D_B.step()

        # Update the learning rate
        lr_scheduler_G.step()
        lr_scheduler_D_B.step()

        # Save models checkpoints
        torch.save(netG_A.state_dict(), 'model/netG_A_pix.pth')
        torch.save(netD_B.state_dict(), 'model/netD_B_pix.pth')

        # Save other checkpoint information
        checkpoint = {'epoch': epoch,
                      'optimizer_G': optimizer_G.state_dict(),
                      'optimizer_D_B': optimizer_D_B.state_dict(),
                      'lr_scheduler_G': lr_scheduler_G.state_dict(),
                      'lr_scheduler_D_B': lr_scheduler_D_B.state_dict()}
        torch.save(checkpoint, 'model/checkpoint.pth')



        # Update the numpy arrays that record the loss
        loss_G_array[epoch] = sum(loss_G_list) / len(loss_G_list)
        loss_D_B_array[epoch] = sum(loss_D_B_list) / len(loss_D_B_list)
        np.savetxt('model/loss_G.txt', loss_G_array)
        np.savetxt('model/loss_D_B.txt', loss_D_B_array)

        end = time.strftime("%H:%M:%S")
        print("current epoch :", epoch, " end time :", end)
        print("G loss :", loss_G_array[epoch], "D_B loss :", loss_D_B_array[epoch])



if __name__ == "__main__":
    main()
