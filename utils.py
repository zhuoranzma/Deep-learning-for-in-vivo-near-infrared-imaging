import torch
from torch.nn import init
import torch.nn as nn
import random
import torchvision.transforms as transforms

# Initialize the weight of the network
def init_weight(net, init_gain = 0.02):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            init.normal_(m.weight.data, 0.0, init_gain)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)
    net.apply(init_func)


# Define the lambda policy for the learning rate decay
class Lambda_rule():
    def __init__(self, start_epoch, initial_epoch, decay_epoch):
        self.start_epoch = start_epoch  #index of the first epoch
        self.initial_epoch = initial_epoch #number of epochs with the initial learning rate
        self.decay_epoch = decay_epoch #number of epochs with learning rate decay

    def step(self, epoch):
        return 1.0 - max(0, epoch + self.start_epoch - self.initial_epoch) / float(self.decay_epoch + 1)

# Set if parameters of a network requires gradient
def set_requires_grad(nets, requires_grad = False):
    if not isinstance(nets, list):
        nets = [nets]
    for net in nets:
        if net is not None:
            for param in net.parameters():
                param.requires_grad = requires_grad

# GAN loss for the network
class GANLoss(nn.Module):
    def __init__(self, target_real_label = 1.0, target_fake_label = 0.0):
        super(GANLoss, self).__init__()
        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))
        self.loss = nn.MSELoss()

    def get_target_tensor(self, prediction, target_is_real):
        # Return a tensor filled with ground-truth label, and has the same size as the prediction
        if target_is_real:
            target_tensor = self.real_label
        else:
            target_tensor = self.fake_label

        return target_tensor.expand_as(prediction)

    def __call__(self, prediction, target_is_real):
        target_tensor = self.get_target_tensor(prediction, target_is_real)
        return self.loss(prediction, target_tensor)

# Store and load previously generated fake images
# The implementation is in reference to https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix.git
class ImagePool():
    def __init__(self, pool_size = 50):
        self.pool_size = pool_size
        if self.pool_size > 0:
            # Create an empty pool
            self.num_imgs = 0
            self.images = []

    def query(self, images):
        # return an image from the image pool
        # If the pool size is 0, just return the input images
        if self.pool_size == 0:
            return images
        return_images = []
        for image in images:
            image = torch.unsqueeze(image.data, 0)
            if self.num_imgs < self.pool_size:
                # If the pool is not full, insert the current image
                self.num_imgs += 1
                self.images.append(image)
                return_images.append(image)
            else:
                p = random.uniform(0, 1)
                if p > 0.5:
                    # return a random image, and insert current image in the pool
                    random_id = random.randint(0, self.pool_size - 1)
                    tmp = self.images[random_id].clone()
                    self.images[random_id] = image
                    return_images.append(tmp)
                else:
                    # return current image
                    return_images.append(image)
        return_images = torch.cat(return_images, 0)
        return return_images

# Transform image tensor to png image
def save_image(tensor, name):
    unloader = transforms.ToPILImage()
    image = tensor.cpu().clone()  # we clone the tensor to not do changes on it
    image = image * 0.5 + 0.5
    image = image.squeeze(0)  # remove the fake batch dimension
    image = unloader(image)
    image.save(name, "PNG")