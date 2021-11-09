import torch
import torchvision
from torch import nn as nn
from models.modules.lpips import dist_model

from utils import util


class GANLoss(nn.Module):
    """Define different GAN objectives.

    The GANLoss class abstracts away the need to create the target label tensor
    that has the same size as the input.
    """

    def __init__(self, gan_mode, target_real_label=1.0, target_fake_label=0.0):
        """ Initialize the GANLoss class.

        Parameters:
            gan_mode (str) - - the type of GAN objective. It currently supports vanilla, lsgan, and wgangp.
            target_real_label (bool) - - label for a real image
            target_fake_label (bool) - - label of a fake image

        Note: Do not use sigmoid as the last layer of Discriminator.
        LSGAN needs no sigmoid. vanilla GANs will handle it with BCEWithLogitsLoss.
        """
        super(GANLoss, self).__init__()
        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))
        self.register_buffer('zero_tensor', torch.tensor(0.))
        self.zero_tensor.requires_grad_(False)
        self.gan_mode = gan_mode
        if gan_mode == 'lsgan':
            self.loss = nn.MSELoss()
        elif gan_mode == 'vanilla':
            self.loss = nn.BCEWithLogitsLoss()
        elif gan_mode == 'wgangp':
            self.loss = None
        elif gan_mode == 'hinge':
            self.loss = None
        else:
            raise NotImplementedError('gan mode %s not implemented' % gan_mode)

    def get_target_tensor(self, prediction, target_is_real):
        if target_is_real:
            target_tensor = self.real_label
        else:
            target_tensor = self.fake_label
        return target_tensor.expand_as(prediction)

    def get_zero_tensor(self, prediction):
        return self.zero_tensor.expand_as(prediction)

    def __call__(self, prediction, target_is_real, for_discriminator=True):
        """Calculate loss given Discriminator's output and grount truth labels.

        Parameters:
            prediction (tensor) - - tpyically the prediction output from a discriminator
            target_is_real (bool) - - if the ground truth label is for real images or fake images

        Returns:
            the calculated loss.
        """
        if self.gan_mode in ['lsgan', 'vanilla']:
            if isinstance(prediction, list):
                losses = []
                for p in prediction:
                    target_tensor = self.get_target_tensor(p, target_is_real)
                    losses.append(self.loss(p, target_tensor))
                return sum(losses)
            else:
                target_tensor = self.get_target_tensor(prediction, target_is_real)
                loss = self.loss(prediction, target_tensor)
        elif self.gan_mode == 'wgangp':
            if target_is_real:
                loss = -prediction.mean()
            else:
                loss = prediction.mean()
        elif self.gan_mode == 'hinge':
            if isinstance(prediction, list):
                loss = 0
                for pred_i in prediction:
                    if isinstance(pred_i, list):
                        pred_i = pred_i[-1]
                    loss_tensor = self(pred_i, target_is_real, for_discriminator)
                    bs = 1 if len(loss_tensor.size()) == 0 else loss_tensor.size(0)
                    new_loss = torch.mean(loss_tensor.view(bs, -1), dim=1)
                    loss += new_loss
                return loss / len(prediction)
            else:
                if for_discriminator:
                    if target_is_real:
                        minval = torch.min(prediction - 1, self.get_zero_tensor(prediction))
                        loss = -torch.mean(minval)
                    else:
                        minval = torch.min(-prediction - 1, self.get_zero_tensor(prediction))
                        loss = -torch.mean(minval)
                else:
                    assert target_is_real
                    loss = -torch.mean(prediction)
        else:
            raise NotImplementedError('gan mode %s not implemented' % self.gan_mode)
        return loss


def cal_gradient_penalty(netD, real_data, fake_data, device, type='mixed', constant=1.0, lambda_gp=10.0):
    """Calculate the gradient penalty loss, used in WGAN-GP paper https://arxiv.org/abs/1704.00028

    Arguments:
        netD (network)              -- discriminator network
        real_data (tensor array)    -- real images
        fake_data (tensor array)    -- generated images from the generator
        device (str)                -- GPU / CPU: from torch.device('cuda:{}'.format(self.gpu_ids[0])) if self.gpu_ids else torch.device('cpu')
        type (str)                  -- if we mix real and fake data or not [real | fake | mixed].
        constant (float)            -- the constant used in formula ( | |gradient||_2 - constant)^2
        lambda_gp (float)           -- weight for this loss

    Returns the gradient penalty loss
    """
    if lambda_gp > 0.0:
        if type == 'real':  # either use real images, fake images, or a linear interpolation of two.
            interpolatesv = real_data
        elif type == 'fake':
            interpolatesv = fake_data
        elif type == 'mixed':
            alpha = torch.rand(real_data.shape[0], 1, device=device)
            alpha = alpha.expand(real_data.shape[0], real_data.nelement() // real_data.shape[0]). \
                contiguous().view(*real_data.shape)
            interpolatesv = alpha * real_data + ((1 - alpha) * fake_data)
        else:
            raise NotImplementedError('{} not implemented'.format(type))
        interpolatesv.requires_grad_(True)
        disc_interpolates = netD(interpolatesv)
        gradients = torch.autograd.grad(outputs=disc_interpolates, inputs=interpolatesv,
                                        grad_outputs=torch.ones(disc_interpolates.size()).to(device),
                                        create_graph=True, retain_graph=True, only_inputs=True)
        gradients = gradients[0].view(real_data.size(0), -1)  # flat the data
        gradient_penalty = (((gradients + 1e-16).norm(2, dim=1) - constant) ** 2).mean() * lambda_gp  # added eps
        return gradient_penalty, gradients
    else:
        return 0.0, None


class VGG19(torch.nn.Module):
    def __init__(self, requires_grad=False):
        super().__init__()
        vgg_pretrained_features = torchvision.models.vgg19(pretrained=True).features
        self.slice1 = torch.nn.Sequential(vgg_pretrained_features[0:2])
        self.slice2 = torch.nn.Sequential(vgg_pretrained_features[2:7])
        self.slice3 = torch.nn.Sequential(vgg_pretrained_features[7:12])
        self.slice4 = torch.nn.Sequential(vgg_pretrained_features[12:21])
        self.slice5 = torch.nn.Sequential(vgg_pretrained_features[21:30])
      
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        h_relu1 = self.slice1(X)
        h_relu2 = self.slice2(h_relu1)
        h_relu3 = self.slice3(h_relu2)
        h_relu4 = self.slice4(h_relu3)
        h_relu5 = self.slice5(h_relu4)
        out = [h_relu1, h_relu2, h_relu3, h_relu4, h_relu5]
        return out


class VGGLoss(nn.Module):
    def __init__(self):
        super(VGGLoss, self).__init__()
        self.vgg = VGG19()
        self.vgg.eval()
        util.set_requires_grad(self.vgg, False)
        self.criterion = nn.L1Loss()
        self.weights = [1.0 / 32, 1.0 / 16, 1.0 / 8, 1.0 / 4, 1.0]

    def forward(self, x, y):
        # x_vgg, y_vgg = self.vgg(x), self.vgg(y)
        # loss = 0
        loss = 0
        x_vgg = self.vgg(x)
        with torch.no_grad():
            y_vgg = self.vgg(y)

        for i in range(len(x_vgg)):
            loss += self.weights[i] * self.criterion(x_vgg[i], y_vgg[i].detach())
        return loss

class PerceptualLoss(torch.nn.Module):
    def __init__(self, model='net-lin', net='alex', colorspace='rgb', spatial=False, use_gpu=True): # VGG using our perceptually-learned weights (LPIPS metric)
    # def __init__(self, model='net', net='vgg', use_gpu=True): # "default" way of using VGG as a perceptual loss
        super(PerceptualLoss, self).__init__()
        print('Setting up Perceptual loss...')
        self.use_gpu = use_gpu
        self.spatial = spatial
        self.model = dist_model.DistModel()
        self.model.initialize(model=model, net=net, use_gpu=use_gpu, colorspace=colorspace, spatial=self.spatial)
        print('...[%s] initialized'%self.model.name())
        print('...Done')

    def forward(self, pred, target, normalize=False):
        """
        Pred and target are Variables.
        If normalize is True, assumes the images are between [0,1] and then scales them between [-1,+1]
        If normalize is False, assumes the images are already between [-1,+1]
        Inputs pred and target are Nx3xHxW
        Output pytorch Variable N long
        """

        if normalize:
            target = 2 * target  - 1
            pred = 2 * pred  - 1

        return self.model.forward(target, pred)

class LPIPSLoss(nn.Module):
    def __init__(self):
        super(LPIPSLoss, self).__init__()
        self.lpips = PerceptualLoss(model='net-lin', net='vgg', use_gpu=True)
        self.lpips.eval()
        util.set_requires_grad(self.lpips, False)
    
    def forward(self, x, y):
        kd_lpips_loss = torch.mean(self.lpips(x, y)) 
        return kd_lpips_loss

    
class VGGLoss_v2(nn.Module):
    def __init__(self):
        super(VGGLoss, self).__init__()
        self.vgg = VGG19()
        self.vgg.eval()
        self.criterion = nn.MSELoss(reduction='sum')
        self.weights = [1.0 / 32, 1.0 / 16, 1.0 / 8, 1.0 / 4, 1.0]

    def forward(self, x, y):
        loss = 0

        with torch.no_grad():
            x_vgg = self.vgg(x)
            y_vgg = self.vgg(y)

        
        for i in range(len(x_vgg)):
            num_fea = x_vgg[i].size(1) # number of feature maps
            h_w_fea = x_vgg[i].size(2) *  x_vgg[i].size(3) # width * height of feature map


            flatten_x = torch.flatten(x_vgg[i].detach(), start_dim=2)
            flatten_y = torch.flatten(y_vgg[i].detach(), start_dim=2)

            gram_x = torch.matmul(flatten_x, flatten_x.transpose(1, 2))
            gram_y = torch.matmul(flatten_y, flatten_y.transpose(1, 2))

            gram_loss = self.criterion(gram_x, gram_y) / (4 * (num_fea**2)  * (h_w_fea**2))
            loss += self.weights[i] * gram_loss
            
        return loss

class R18(torch.nn.Module):
    def __init__(self, requires_grad=False):
        super().__init__()
        resnet = torchvision.models.resnet18(pretrained=True)
        self.slice1 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool, resnet.layer1)
        self.slice2 = resnet.layer2
        self.slice3 = resnet.layer3
        self.slice4 = resnet.layer4
        
        if not requires_grad:
            for param in resnet.parameters():
                param.requires_grad = False

    def forward(self, X):
        h_relu1 = self.slice1(X)
        h_relu2 = self.slice2(h_relu1)
        h_relu3 = self.slice3(h_relu2)
        h_relu4 = self.slice4(h_relu3)
        out = [h_relu1, h_relu2, h_relu3, h_relu4]
        return out

    
class R18Loss(nn.Module):
    def __init__(self):
        super(R18Loss, self).__init__()
        self.resnet = R18()
        self.resnet.eval()
        for param in self.resnet.parameters():
            param.requires_grad = False
        self.criterion = nn.L1Loss()
        self.weights = [1.0 / 16, 1.0 / 8, 1.0 / 4, 1.0]

    def forward(self, x, y):
        loss = 0
        x_res = self.resnet(x)
      
        with torch.no_grad():
            y_res = self.resnet(y)

        for i in range(len(x_res)):
            loss += self.weights[i] * self.criterion(x_res[i], y_res[i].detach())
        return loss
