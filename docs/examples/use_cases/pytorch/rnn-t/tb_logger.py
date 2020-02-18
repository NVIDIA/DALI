import torch.utils.tensorboard as tb

class DummyLogger:
    def log_scalar(*args, **kwargs):
        pass

    def log_params(*args, **kwargs):
        pass

    def log_grad(*args, **kwargs):
        pass

    def train_end(*args, **kwargs):
        pass


class TensorBoardLogger(DummyLogger):
    def __init__(self, path, model, histogram=False):
        self.writer = tb.SummaryWriter(log_dir=str(path))
        self.model = model
        self.histogram = histogram

    def log_scalar(self, name, value, step, stage='train'):
        self.writer.add_scalar(
            f'{stage}/{name}',
            value,
            global_step=step
        )

    def log_grad(self, step):
        if not self.histogram:
            return
        for name, param in self.model.named_parameters():
            if param.grad is not None:
                self.writer.add_histogram(
                    name.replace('.', '/'),
                    param.grad,
                    global_step=step
                )

    def log_params(self, step):
        if not self.histogram:
            return
        for name, param in self.model.named_parameters():
            self.writer.add_histogram(
                name.replace('.', '/'),
                param,
                global_step=step
            )

    def train_end(self):
        self.writer.close()
