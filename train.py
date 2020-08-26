import networks
import argparse
import utils
import torch
import itertools
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from datasets import ImageDataset
import time
import numpy as np



# Get training options from the command line
def get_opt():
    parser = argparse.ArgumentParser()
    # Training parameters
    parser.add_argument('--epoch', type=int, default=0, help='starting epoch')
    parser.add_argument('--n_epochs', type = int, default = 100, help = 'number of epochs with initial learning rate')
    parser.add_argument('--n_epochs_decay', type = int, default = 100, help = 'number of epochs starting the decay of learning rate')
    parser.add_argument('--beta1', type = float, default = 0.5, help = 'momentum term of the Adam optimizer')
    parser.add_argument('--lr', type = float, default = 0.0002, help = 'initial learning rate')
    parser.add_argument('--batch_size', type = int, default = 1, help = 'batch size of training')
    parser.add_argument('--cuda', action='store_true', help='use GPU computation')
    parser.add_argument('--rootdir', type=str, default='datasets/NIRI_to_NIRII/', help='root directory of the dataset')
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
    parser.add_argument('--cycle_loss', type = float, default=10, help = 'coefficient of cycle consistent loss')
    parser.add_argument('--identity_loss', type = float, default=0, help = 'coefficient of identity loss')

    opt = parser.parse_args()
    return opt


def main():
    # Get training options
    opt = get_opt()

    # Define the networks
    # netG_A: used to transfer image from domain A to domain B
    # netG_B: used to transfer image from domain B to domain A
    netG_A = networks.Generator(opt.input_nc, opt.output_nc, opt.ngf, opt.n_res, opt.dropout)
    netG_B = networks.Generator(opt.output_nc, opt.input_nc, opt.ngf, opt.n_res, opt.dropout)
    if opt.u_net:
        netG_A = networks.U_net(opt.input_nc, opt.output_nc, opt.ngf)
        netG_B = networks.U_net(opt.output_nc, opt.input_nc, opt.ngf)

    # netD_A: used to test whether an image is from domain B
    # netD_B: used to test whether an image is from domain A
    netD_A = networks.Discriminator(opt.input_nc, opt.ndf)
    netD_B = networks.Discriminator(opt.output_nc, opt.ndf)

    # Initialize the networks
    if opt.cuda:
        netG_A.cuda()
        netG_B.cuda()
        netD_A.cuda()
        netD_B.cuda()
    utils.init_weight(netG_A)
    utils.init_weight(netG_B)
    utils.init_weight(netD_A)
    utils.init_weight(netD_B)

    if opt.pretrained:
        netG_A.load_state_dict(torch.load('pretrained/netG_A.pth'))
        netG_B.load_state_dict(torch.load('pretrained/netG_B.pth'))
        netD_A.load_state_dict(torch.load('pretrained/netD_A.pth'))
        netD_B.load_state_dict(torch.load('pretrained/netD_B.pth'))


    # Define the loss functions
    criterion_GAN = utils.GANLoss()
    if opt.cuda:
        criterion_GAN.cuda()

    criterion_cycle = torch.nn.L1Loss()
    # Alternatively, can try MSE cycle consistency loss
    #criterion_cycle = torch.nn.MSELoss()
    criterion_identity = torch.nn.L1Loss()

    # Define the optimizers
    optimizer_G = torch.optim.Adam(itertools.chain(netG_A.parameters(), netG_B.parameters()), lr=opt.lr, betas=(opt.beta1, 0.999))
    optimizer_D_A = torch.optim.Adam(netD_A.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
    optimizer_D_B = torch.optim.Adam(netD_B.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))

    # Create learning rate schedulers
    lr_scheduler_G = torch.optim.lr_scheduler.LambdaLR(optimizer_G, lr_lambda = utils.Lambda_rule(opt.epoch, opt.n_epochs, opt.n_epochs_decay).step)
    lr_scheduler_D_A = torch.optim.lr_scheduler.LambdaLR(optimizer_D_A, lr_lambda = utils.Lambda_rule(opt.epoch, opt.n_epochs, opt.n_epochs_decay).step)
    lr_scheduler_D_B = torch.optim.lr_scheduler.LambdaLR(optimizer_D_B, lr_lambda = utils.Lambda_rule(opt.epoch, opt.n_epochs, opt.n_epochs_decay).step)

    Tensor = torch.cuda.FloatTensor if opt.cuda else torch.Tensor
    input_A = Tensor(opt.batch_size, opt.input_nc, opt.sizeh, opt.sizew)
    input_B = Tensor(opt.batch_size, opt.output_nc, opt.sizeh, opt.sizew)

    # Define two image pools to store generated images
    fake_A_pool = utils.ImagePool()
    fake_B_pool = utils.ImagePool()

    # Define the transform, and load the data
    transform = transforms.Compose([transforms.Resize((opt.sizeh, opt.sizew)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,))])
    dataloader = DataLoader(ImageDataset(opt.rootdir, transform = transform, mode = 'train'), batch_size=opt.batch_size, shuffle=True, num_workers=opt.n_cpu)

    # numpy arrays to store the loss of epoch
    loss_G_array = np.zeros(opt.n_epochs + opt.n_epochs_decay)
    loss_D_A_array = np.zeros(opt.n_epochs + opt.n_epochs_decay)
    loss_D_B_array = np.zeros(opt.n_epochs + opt.n_epochs_decay)

    # Training
    for epoch in range(opt.epoch, opt.n_epochs + opt.n_epochs_decay):
        start = time.strftime("%H:%M:%S")
        print("current epoch :", epoch, " start time :", start)
        # Empty list to store the loss of each mini-batch
        loss_G_list = []
        loss_D_A_list = []
        loss_D_B_list = []

        for i, batch in enumerate(dataloader):
            if i % 50 == 1:
                print("current step: ", i)
                current = time.strftime("%H:%M:%S")
                print("current time :", current)
                print("last loss G:", loss_G_list[-1], "last loss D_A", loss_D_A_list[-1], "last loss D_B", loss_D_B_list[-1])
            real_A = input_A.copy_(batch['A'])
            real_B = input_B.copy_(batch['B'])

            # Train the generator
            optimizer_G.zero_grad()

            # Compute fake images and reconstructed images
            fake_B = netG_A(real_A)
            fake_A = netG_B(real_B)

            if opt.identity_loss != 0:
                same_B = netG_A(real_B)
                same_A = netG_B(real_A)

            # discriminators require no gradients when optimizing generators
            utils.set_requires_grad([netD_A, netD_B], False)

            # Identity loss
            if opt.identity_loss != 0:
                loss_identity_A = criterion_identity(same_A, real_A) * opt.identity_loss
                loss_identity_B = criterion_identity(same_B, real_B) * opt.identity_loss

            # GAN loss
            prediction_fake_B = netD_B(fake_B)
            loss_gan_B = criterion_GAN(prediction_fake_B, True)
            prediction_fake_A = netD_A(fake_A)
            loss_gan_A = criterion_GAN(prediction_fake_A, True)

            # Cycle consistent loss
            recA = netG_B(fake_B)
            recB = netG_A(fake_A)
            loss_cycle_A = criterion_cycle(recA, real_A) * opt.cycle_loss
            loss_cycle_B = criterion_cycle(recB, real_B) * opt.cycle_loss

            # total loss without the identity loss
            loss_G = loss_gan_B + loss_gan_A + loss_cycle_A + loss_cycle_B

            if opt.identity_loss != 0:
                loss_G += loss_identity_A + loss_identity_B

            loss_G_list.append(loss_G.item())
            loss_G.backward()
            optimizer_G.step()

            # Train the discriminator
            utils.set_requires_grad([netD_A, netD_B], True)


            # Train the discriminator D_A
            optimizer_D_A.zero_grad()
            # real images
            pred_real = netD_A(real_A)
            loss_D_real = criterion_GAN(pred_real, True)

            # fake images
            fake_A = fake_A_pool.query(fake_A)
            pred_fake = netD_A(fake_A.detach())
            loss_D_fake = criterion_GAN(pred_fake, False)

            #total loss
            loss_D_A = (loss_D_real + loss_D_fake) * 0.5
            loss_D_A_list.append(loss_D_A.item())
            loss_D_A.backward()
            optimizer_D_A.step()

            # Train the discriminator D_B
            optimizer_D_B.zero_grad()
            # real images
            pred_real = netD_B(real_B)
            loss_D_real = criterion_GAN(pred_real, True)

            # fake images
            fake_B = fake_B_pool.query(fake_B)
            pred_fake = netD_B(fake_B.detach())
            loss_D_fake = criterion_GAN(pred_fake, False)

            # total loss
            loss_D_B = (loss_D_real + loss_D_fake) * 0.5
            loss_D_B_list.append(loss_D_B.item())
            loss_D_B.backward()
            optimizer_D_B.step()

        # Update the learning rate
        lr_scheduler_G.step()
        lr_scheduler_D_A.step()
        lr_scheduler_D_B.step()

        # Save models checkpoints
        torch.save(netG_A.state_dict(), 'model/netG_A.pth')
        torch.save(netG_B.state_dict(), 'model/netG_B.pth')
        torch.save(netD_A.state_dict(), 'model/netD_A.pth')
        torch.save(netD_B.state_dict(), 'model/netD_B.pth')




        # Save other checkpoint information
        checkpoint = {'epoch': epoch,
                      'optimizer_G': optimizer_G.state_dict(),
                      'optimizer_D_A': optimizer_D_A.state_dict(),
                      'optimizer_D_B': optimizer_D_B.state_dict(),
                      'lr_scheduler_G': lr_scheduler_G.state_dict(),
                      'lr_scheduler_D_A': lr_scheduler_D_A.state_dict(),
                      'lr_scheduler_D_B': lr_scheduler_D_B.state_dict()}
        torch.save(checkpoint, 'model/checkpoint.pth')



        # Update the numpy arrays that record the loss
        loss_G_array[epoch] = sum(loss_G_list) / len(loss_G_list)
        loss_D_A_array[epoch] = sum(loss_D_A_list) / len(loss_D_A_list)
        loss_D_B_array[epoch] = sum(loss_D_B_list) / len(loss_D_B_list)
        np.savetxt('model/loss_G.txt', loss_G_array)
        np.savetxt('model/loss_D_A.txt', loss_D_A_array)
        np.savetxt('model/loss_D_b.txt', loss_D_B_array)


        if epoch % 10 == 9:
            torch.save(netG_A.state_dict(), 'model/netG_A' + str(epoch) + '.pth')
            torch.save(netG_B.state_dict(), 'model/netG_B' + str(epoch) + '.pth')
            torch.save(netD_A.state_dict(), 'model/netD_A' + str(epoch) + '.pth')
            torch.save(netD_B.state_dict(), 'model/netD_B' + str(epoch) + '.pth')

        end = time.strftime("%H:%M:%S")
        print("current epoch :", epoch, " end time :", end)
        print("G loss :", loss_G_array[epoch], "D_A loss :", loss_D_A_array[epoch], "D_B loss :", loss_D_B_array[epoch])



if __name__ == "__main__":
    main()

