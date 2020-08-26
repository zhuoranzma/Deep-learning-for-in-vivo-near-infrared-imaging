import networks
import argparse
import utils
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from datasets import ImageDataset
from PIL import Image


# Get the options for testing
def get_opt():
    parser = argparse.ArgumentParser()
    # Parameters for testing
    parser.add_argument('--batch_size', type=int, default=1, help='batch size of testing')
    parser.add_argument('--cuda', action='store_true', help='use GPU computation')
    parser.add_argument('--rootdir', type=str, default='datasets/NIRI_to_NIRII/', help='root directory of the dataset')
    parser.add_argument('--n_cpu', type=int, default=4, help='number of cpu threads to use during batch generation')
    parser.add_argument('--u_net', action='store_true', help='use U-net generator')

    # Model parameters
    parser.add_argument('--sizeh', type=int, default=512, help='size of the image')
    parser.add_argument('--sizew', type=int, default=640, help='size of the image')
    parser.add_argument('--input_nc', type=int, default=1, help='number of input channels')
    parser.add_argument('--output_nc', type=int, default=1, help='number of output channels')
    parser.add_argument('--ngf', type=int, default=64, help='number of filters in the generator')
    parser.add_argument('--ndf', type=int, default=64, help='number of filters in the discriminator')
    parser.add_argument('--dropout', type=bool, default=False, help='whether to use dropout')
    parser.add_argument('--n_res', type=int, default=9, help='number of resNet blocks')
    parser.add_argument('--net_GA', type=str, default='model/netG_A.pth', help='path of the parameters of the generator A')

    opt = parser.parse_args()
    return opt


def main():
    opt = get_opt()

    # Define the Generators, only G_A is used for testing
    netG_A = networks.Generator(opt.input_nc, opt.output_nc, opt.ngf, opt.n_res, opt.dropout)
    if opt.u_net:
        netG_A = networks.U_net(opt.input_nc, opt.output_nc, opt.ngf)

    if opt.cuda:
        netG_A.cuda()
    # Do not need to track the gradients during testing
    utils.set_requires_grad(netG_A, False)
    netG_A.eval()
    netG_A.load_state_dict(torch.load(opt.net_GA))

    # Load the data
    transform = transforms.Compose([transforms.Resize((opt.sizeh, opt.sizew)),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.5,), (0.5,))])
    dataloader = DataLoader(ImageDataset(opt.rootdir, transform=transform, mode='val'), batch_size=opt.batch_size,
                            shuffle=False, num_workers=opt.n_cpu)

    Tensor = torch.cuda.FloatTensor if opt.cuda else torch.Tensor
    input_A = Tensor(opt.batch_size, opt.input_nc, opt.size, opt.size)

    for i, batch in enumerate(dataloader):
        name, image = batch
        real_A = input_A.copy_(image)
        fake_B = netG_A(real_A)
        batch_size = len(name)
        # Save the generated images
        for j in range(batch_size):
            image_name = name[j].split('/')[-1]
            path = 'generated_image/' + image_name
            utils.save_image(fake_B[j, :, :, :], path)

if __name__ == '__main__':
    main()
