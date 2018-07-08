import torch
import torch.nn as nn
import torch.nn.functional as F

class Mixup(object):
    def __init__(self, alpha, device: torch.device, mix_triplets=False):
        self.alpha = alpha
        self.beta = torch.distributions.Beta(self.alpha, self.alpha)
        self.device = device
        self.mix_triplets = mix_triplets
    
    def __call__(self, images, targets):
        perm = torch.randperm(images.size(0))
        perm_images = images[perm]
        perm_targets = targets[perm]
        lam = self.beta.sample((images.size(0),)).to(self.device)
        if self.mix_triplets:
            return (lam.view(-1,1,1,1,1)*images + (1-lam).view(-1,1,1,1,1)*perm_images), (lam*targets + (1-lam)*perm_targets)
        else:
            return (lam.view(-1,1,1,1)*images + (1-lam).view(-1,1,1,1)*perm_images), (lam*targets + (1-lam)*perm_targets)

class BinaryCrossEntropy(nn.Module):
    def __init__(self, device: torch.device, weight=torch.tensor([1., 1.]), size_average=True):
        super(BinaryCrossEntropy, self).__init__()
        self.size_average = size_average
        self.one = torch.tensor(1.).to(device)
        self.weight = weight.to(device)
    
    def forward(self, inputs, targets):
        inputs = F.sigmoid(inputs)

        if self.size_average:
            return torch.mean(- torch.log(inputs) * targets * self.weight[1] - (self.one - targets) * torch.log(self.one - inputs) * self.weight[0])
        else:
            return torch.sum(- torch.log(inputs) * targets * self.weight[1] - (self.one - targets) * torch.log(self.one - inputs) * self.weight[0])