import torch
import torch.nn as nn
import torch.nn.functional as F

class Mixup(object):
    def __init__(self, alpha, device: torch.device, mix_triplets=False):
        self.alpha = alpha
        self.beta = torch.distributions.Beta(self.alpha, self.alpha)
        self.device = device
        self.one = torch.tensor(1.).to(device)
    
    def __call__(self, images, targets):
        perm = torch.randperm(images.size(0))
        perm_images = images[perm]
        perm_targets = targets[perm]
        lam = self.beta.sample((images.size(0),)).to(self.device)
        return (lam.view(-1,1,1,1)*images + (self.one-lam).view(-1,1,1,1)*perm_images), (lam.unsqueeze(-1)*targets + (self.one-lam).unsqueeze(-1)*perm_targets)

class OneHotCrossEntropy(nn.Module):
    def __init__(self, device: torch.device):
        super(OneHotCrossEntropy, self).__init__()
        self.one = torch.tensor(1.).to(device)
    
    def forward(self, inputs, targets):
        inputs = F.log_softmax(inputs, dim=1)

        return torch.sum(- inputs * targets) / targets.size(0)
