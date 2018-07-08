import math
from torch.optim.lr_scheduler import _LRScheduler


class SnapScheduler(_LRScheduler):
    def __init__(self, optimizer, num_epochs, num_snaps, init_lr, last_epoch=-1):
        """
        Args:
            optimizer (Optimizer): Wrapped optimizer.
            num_epochs (int): Number of epochs.
            num_snaps (int): Number of snaps to be saved.
            init_lr (float): Initial learning rate.
            last_epoch (int): The index of last epoch. Default: -1.
        """
        self.T = num_epochs
        self.M = num_snaps
        self.init_lr = init_lr
        self.__new_lr = init_lr
        super(SnapScheduler, self).__init__(optimizer, last_epoch)
    
    def get_lr(self):
        """
            ArXiv 1704.00109, formula (2):
                alpha(t) = alpha_0 / 2  *  ( cos( (pi * (t-1) % (T / M) ) / (T / M) ) + 1 )
                (T / M) is integer division rounded up
        """
        _inner = math.pi * (self.last_epoch % (self.T // self.M)) / (self.T // self.M)
        self.__new_lr = self.init_lr / 2 * (math.cos(_inner) + 1)
        print("New learning rate:", self.__new_lr)
        return [self.__new_lr for _ in self.base_lrs]
    
    def save_model(self, epoch) -> bool:
        if (epoch % (self.T // self.M)) == 0:
            return True  
        else:
            return False