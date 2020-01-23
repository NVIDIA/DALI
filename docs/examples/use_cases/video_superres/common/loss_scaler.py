import torch

class LossScaler:

    def __init__(self, scale=1):
        self.cur_scale = scale

    # `params` is a list / generator of torch.Variable
    def has_overflow(self, params):
        return False

    # `x` is a torch.Tensor
    def _has_inf_or_nan(x):
        return False

    # `overflow` is boolean indicating whether we overflowed in gradient
    def update_scale(self, overflow):
        pass

    @property
    def loss_scale(self):
        return self.cur_scale

    def scale_gradient(self, module, grad_in, grad_out):
        return tuple(self.loss_scale * g for g in grad_in)

    def backward(self, loss):
        scaled_loss = loss*self.loss_scale
        scaled_loss.backward()

class DynamicLossScaler:

    def __init__(self,
                 init_scale=2**32,
                 scale_factor=2.,
                 scale_window=1000):
        self.cur_scale = init_scale
        self.cur_iter = 0
        self.last_overflow_iter = -1
        self.scale_factor = scale_factor
        self.scale_window = scale_window

    # `params` is a list / generator of torch.Variable
    def has_overflow(self, params):
#        return False
        for p in params:
            if p.grad is not None and DynamicLossScaler._has_inf_or_nan(p.grad.data):
                return True

        return False

    # `x` is a torch.Tensor
    def _has_inf_or_nan(x):
        inf_count = torch.sum(x.abs() == float('inf'))
        if inf_count > 0:
            return True
        nan_count = torch.sum(x != x)
        return nan_count > 0

    # `overflow` is boolean indicating whether we overflowed in gradient
    def update_scale(self, overflow):
        if overflow:
            #self.cur_scale /= self.scale_factor
            self.cur_scale = max(self.cur_scale/self.scale_factor, 1)
            self.last_overflow_iter = self.cur_iter
        else:
            if (self.cur_iter - self.last_overflow_iter) % self.scale_window == 0:
                self.cur_scale *= self.scale_factor
#        self.cur_scale = 1
        self.cur_iter += 1

    @property
    def loss_scale(self):
        return self.cur_scale

    def scale_gradient(self, module, grad_in, grad_out):
        return tuple(self.loss_scale * g for g in grad_in)

    def backward(self, loss):
        scaled_loss = loss*self.loss_scale
        scaled_loss.backward()
