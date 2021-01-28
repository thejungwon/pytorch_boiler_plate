from torch.optim.optimizer import Optimizer, required
import copy

class MyOptimizer(Optimizer):

    def __init__(self, params):
        defaults = {}
        super(MyOptimizer, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(MyOptimizer, self).__setstate__(state)

    def step(self, closure=None):
        
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            pass

        return loss