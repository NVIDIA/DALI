import torch
from torch import nn
from torch.autograd import Variable
from torch.nn.parameter import Parameter
from torch._utils import _flatten_dense_tensors, _unflatten_dense_tensors

from common.loss_scaler import DynamicLossScaler, LossScaler

FLOAT_TYPES = (torch.FloatTensor, torch.cuda.FloatTensor)
HALF_TYPES = (torch.HalfTensor, torch.cuda.HalfTensor)

def conversion_helper(val, conversion):
    """Apply conversion to val. Recursively apply conversion if `val` is a nested tuple/list structure."""
    if not isinstance(val, (tuple, list)):
        return conversion(val)
    rtn =  [conversion_helper(v, conversion) for v in val]
    if isinstance(val, tuple):
        rtn = tuple(rtn)
    return rtn

def fp32_to_fp16(val):
    """Convert fp32 `val` to fp16"""
    def half_conversion(val):
        val_typecheck = val
        if isinstance(val_typecheck, (Parameter, Variable)):
            val_typecheck = val.data
        if isinstance(val_typecheck, FLOAT_TYPES):
            val = val.half()
        return val
    return conversion_helper(val, half_conversion)

def fp16_to_fp32(val):
    """Convert fp16 `val` to fp32"""
    def float_conversion(val):
        val_typecheck = val
        if isinstance(val_typecheck, (Parameter, Variable)):
            val_typecheck = val.data
        if isinstance(val_typecheck, HALF_TYPES):
            val = val.float()
        return val
    return conversion_helper(val, float_conversion)

class FP16_Module(nn.Module):
    def __init__(self, module):
        super(FP16_Module, self).__init__()
        self.add_module('module', module.half())

    def forward(self, *inputs, **kwargs):
        return fp16_to_fp32(self.module(*(fp32_to_fp16(inputs)), **kwargs))

class FP16_Optimizer(object):
    """
    FP16_Optimizer is designed to wrap an existing PyTorch optimizer,
    and enable an fp16 model to be trained using a main copy of fp32 weights.

    Args:
        optimizer (torch.optim.optimizer):  Existing optimizer containing initialized fp16 parameters.  Internally, FP16_Optimizer replaces the passed optimizer's fp16 parameters with new fp32 parameters copied from the original ones.  FP16_Optimizer also stores references to the original fp16 parameters, and updates these fp16 parameters from the main fp32 copy after each step.
        static_loss_scale (float, optional, default=1.0):  Loss scale used internally to scale fp16 gradients computed by the model.  Scaled gradients will be copied to fp32, then downscaled before being applied to the fp32 main params, so static_loss_scale should not affect learning rate.
        dynamic_loss_scale (bool, optional, default=False):  Use dynamic loss scaling.  If True, this will override any static_loss_scale option.

    """

    def __init__(self, optimizer, static_loss_scale=1.0, dynamic_loss_scale=False):
        if not torch.cuda.is_available:
            raise SystemError('Cannot use fp16 without CUDA')

        self.fp16_param_groups = []
        self.fp32_param_groups = []
        self.fp32_flattened_groups = []
        for i, param_group in enumerate(optimizer.param_groups):
            print("FP16_Optimizer processing param group {}:".format(i))
            fp16_params_this_group = []
            fp32_params_this_group = []
            for param in param_group['params']:
                if param.requires_grad:
                    if param.type() == 'torch.cuda.HalfTensor':
                        print("FP16_Optimizer received torch.cuda.HalfTensor with {}"
                              .format(param.size()))
                        fp16_params_this_group.append(param)
                    elif param.type() == 'torch.cuda.FloatTensor':
                        print("FP16_Optimizer received torch.cuda.FloatTensor with {}"
                              .format(param.size()))
                        fp32_params_this_group.append(param)
                    else:
                        raise TypeError("Wrapped parameters must be either "
                                        "torch.cuda.FloatTensor or torch.cuda.HalfTensor. "
                                        "Received {}".format(param.type()))

            fp32_flattened_this_group = None
            if len(fp16_params_this_group) > 0:
                fp32_flattened_this_group = _flatten_dense_tensors(
                    [param.detach().data.clone().float() for param in fp16_params_this_group])

                fp32_flattened_this_group = Variable(fp32_flattened_this_group, requires_grad = True)

                fp32_flattened_this_group.grad = fp32_flattened_this_group.new(
                    *fp32_flattened_this_group.size())

            # python's lovely list concatenation via +
            if fp32_flattened_this_group is not None:
                param_group['params'] = [fp32_flattened_this_group] + fp32_params_this_group
            else:
                param_group['params'] = fp32_params_this_group

            self.fp16_param_groups.append(fp16_params_this_group)
            self.fp32_param_groups.append(fp32_params_this_group)
            self.fp32_flattened_groups.append(fp32_flattened_this_group)

        # print("self.fp32_flattened_groups = ", self.fp32_flattened_groups)
        # print("self.fp16_param_groups = ", self.fp16_param_groups)

        self.optimizer = optimizer.__class__(optimizer.param_groups)

        # self.optimizer.load_state_dict(optimizer.state_dict())

        self.param_groups = self.optimizer.param_groups

        if dynamic_loss_scale:
            self.dynamic_loss_scale = True
            self.loss_scaler = DynamicLossScaler()
        else:
            self.dynamic_loss_scale = False
            self.loss_scaler = LossScaler(static_loss_scale)

        self.overflow = False
        self.first_closure_call_this_step = True

    def zero_grad(self):
        """
        Zero fp32 and fp16 parameter grads.
        """
        self.optimizer.zero_grad()
        for fp16_group in self.fp16_param_groups:
            for param in fp16_group:
                if param.grad is not None:
                    param.grad.detach_()
                    param.grad.zero_()

    def _check_overflow(self):
        params = []
        for group in self.fp16_param_groups:
            for param in group:
                params.append(param)
        for group in self.fp32_param_groups:
            for param in group:
                params.append(param)
        self.overflow = self.loss_scaler.has_overflow(params)

    def _update_scale(self, has_overflow=False):
        self.loss_scaler.update_scale(has_overflow)

    def _copy_grads_fp16_to_fp32(self):
        for fp32_group, fp16_group in zip(self.fp32_flattened_groups, self.fp16_param_groups):
            if len(fp16_group) > 0:
                # This might incur one more deep copy than is necessary.
                fp32_group.grad.data.copy_(
                    _flatten_dense_tensors([fp16_param.grad.data for fp16_param in fp16_group]))

    def _downscale_fp32(self):
        if self.loss_scale != 1.0:
            for param_group in self.optimizer.param_groups:
                for param in param_group['params']:
                    param.grad.data.mul_(1./self.loss_scale)

    def clip_fp32_grads(self, max_norm, norm_type=2):
        """
        Clips fp32 main gradients via torch.nn.utils.clip_grad_norm.

        Args:
            max_norm (float or int): max norm of the gradients
            norm_type (float or int): type of the used p-norm. Can be ``'inf'`` for
                infinity norm.

        Returns:
            Total norm of the current fp32 gradients (viewed as a single vector).

        .. warning::
            Returns -1 if the most recently computed fp16 gradients overflowed (that is, if self.overflow is True).
        """
        if not self.overflow:
            fp32_params = []
            for param_group in self.optimizer.param_groups:
                for param in param_group['params']:
                    fp32_params.append(param)
            return torch.nn.utils.clip_grad_norm(fp32_params, max_norm, norm_type)
        else:
            return -1

    def _copy_params_fp32_to_fp16(self):
        for fp16_group, fp32_group in zip(self.fp16_param_groups, self.fp32_flattened_groups):
            if len(fp16_group) > 0:
                for fp16_param, fp32_data in zip(fp16_group,
                    _unflatten_dense_tensors(fp32_group.data, fp16_group)):
                    fp16_param.data.copy_(fp32_data)

    def state_dict(self):
        """
        Returns a dict containing the current state of this FP16_Optimizer instance.
        This dict contains attributes of FP16_Optimizer, as well as the state_dict
        of the contained PyTorch optimizer.

        Untested.
        """
        state_dict = {}
        state_dict['loss_scaler'] = self.loss_scaler
        state_dict['dynamic_loss_scale'] = self.dynamic_loss_scale
        state_dict['overflow'] = self.overflow
        state_dict['first_closure_call_this_step'] = self.first_closure_call_this_step
        state_dict['optimizer_state_dict'] = self.optimizer.state_dict()
        return state_dict

    def load_state_dict(self, state_dict):
        """
        Loads a state_dict created by an earlier call to state_dict.

        Untested.
        """
        self.loss_scaler = state_dict['loss_scaler']
        self.dynamic_loss_scale = state_dict['dynamic_loss_scale']
        self.overflow = state_dict['overflow']
        self.first_closure_call_this_step = state_dict['first_closure_call_this_step']
        self.optimizer.load_state_dict(state_dict['optimizer_state_dict'])

    def step(self, closure=None): # could add clip option.
        """
        If no closure is supplied, step should be called after fp16_optimizer_obj.backward(loss).
        step updates the fp32 main copy of parameters using the optimizer supplied to
        FP16_Optimizer's constructor, then copies the updated fp32 params into the fp16 params
        originally referenced by Fp16_Optimizer's constructor, so the user may immediately run
        another forward pass using their model.

        If a closure is supplied, step may be called without a prior call to self.backward(loss).
        However, the user should take care that any loss.backward() call within the closure
        has been replaced by fp16_optimizer_obj.backward(loss).

        Args:
           closure (optional):  Closure that will be supplied to the underlying optimizer originally passed to FP16_Optimizer's constructor.  closure should call zero_grad on the FP16_Optimizer object, compute the loss, call .backward(loss), and return the loss.

        Closure example::

            # optimizer is assumed to be an FP16_Optimizer object, previously constructed from an
            # existing pytorch optimizer.
            for input, target in dataset:
                def closure():
                    optimizer.zero_grad()
                    output = model(input)
                    loss = loss_fn(output, target)
                    optimizer.backward(loss)
                    return loss
                optimizer.step(closure)

        .. note::
            The only changes that need to be made compared to
            `ordinary optimizer closures`_ are that "optimizer" itself should be an instance of
            FP16_Optimizer, and that the call to loss.backward should be replaced by
            optimizer.backward(loss).

        .. warning::
            Currently, calling step with a closure is not compatible with dynamic loss scaling.

        .. _`ordinary optimizer closures`:
            http://pytorch.org/docs/master/optim.html#optimizer-step-closure
        """
        if closure is not None and isinstance(self.loss_scaler, DynamicLossScaler):
            raise TypeError("Using step with a closure is currently not "
                            "compatible with dynamic loss scaling.")

        scale = self.loss_scaler.loss_scale
        self._update_scale(self.overflow)

        if self.overflow:
            print("OVERFLOW! Skipping step. Attempted loss scale: {}".format(scale))
            return

        if closure is not None:
            self._step_with_closure(closure)
        else:
            self.optimizer.step()

        self._copy_params_fp32_to_fp16()

        return

    def _step_with_closure(self, closure):
        def wrapped_closure():
            if self.first_closure_call_this_step:
                """
                We expect that the fp16 params are initially fresh on entering self.step(),
                so _copy_params_fp32_to_fp16() is unnecessary the first time wrapped_closure()
                is called within self.optimizer.step().
                """
                self.first_closure_call_this_step = False
            else:
                """
                If self.optimizer.step() internally calls wrapped_closure more than once,
                it may update the fp32 params after each call.  However, self.optimizer
                doesn't know about the fp16 params at all.  If the fp32 params get updated,
                we can't rely on self.optimizer to refresh the fp16 params.  We need
                to handle that manually:
                """
                self._copy_params_fp32_to_fp16()

            """
            Our API expects the user to give us ownership of the backward() call by
            replacing all calls to loss.backward() with optimizer.backward(loss).
            This requirement holds whether or not the call to backward() is made within
            a closure.
            If the user is properly calling optimizer.backward(loss) within "closure,"
            calling closure() here will give the fp32 main params fresh gradients
            for the optimizer to play with,
            so all wrapped_closure needs to do is call closure() and return the loss.
            """
            temp_loss = closure()
            return temp_loss

        self.optimizer.step(wrapped_closure)

        self.first_closure_call_this_step = True

    def backward(self, loss, update_fp32_grads=True):
        """
        fp16_optimizer_obj.backward performs the following conceptual operations:

        fp32_loss = loss.float() (see first Note below)

        scaled_loss = fp32_loss*loss_scale

        scaled_loss.backward(), which accumulates scaled gradients into the .grad attributes of the
        fp16 model's leaves.

        fp16 grads are then copied to the stored fp32 params' .grad attributes (see second Note).

        Finally, fp32 grads are divided by loss_scale.

        In this way, after fp16_optimizer_obj.backward, the fp32 parameters have fresh gradients,
        and fp16_optimizer_obj.step may be called.

        .. note::
            Converting the loss to fp32 before applying the loss scale provides some
            additional safety against overflow if the user has supplied an fp16 value.
            However, for maximum overflow safety, the user should
            compute the loss criterion (MSE, cross entropy, etc) in fp32 before supplying it to
            fp16_optimizer_obj.backward.

        .. note::
            The gradients found in an fp16 model's leaves after a call to
            fp16_optimizer_obj.backward should not be regarded as valid in general,
            because it's possible
            they have been scaled (and in the case of dynamic loss scaling,
            the scale factor may change over time).
            If the user wants to inspect gradients after a call to fp16_optimizer_obj.backward,
            only the main gradients should be regarded as valid, and can be retrieved via
            :attr:`inspect_fp32_grad_data()`.


        Args:
            loss:  The loss output by the user's model.  loss may be either float or half (but see first Note above).
            update_fp32_grads (bool, optional, default=True):  Option to copy fp16 grads to fp32 grads on this call.  By setting this to False, the user can delay this copy, which is useful to eliminate redundant fp16->fp32 grad copies if fp16_optimizer_obj.backward is being called on multiple losses in one iteration.  If set to False, the user becomes responsible for calling fp16_optimizer_obj.update_fp32_grads before calling fp16_optimizer_obj.step.

        Example::

            # Ordinary operation:
            optimizer.backward(loss)

            # Naive operation with multiple losses (technically valid, but less efficient):
            # fp32 grads will be correct after the second call,  but
            # the first call incurs an unnecessary fp16->fp32 grad copy.
            optimizer.backward(loss1)
            optimizer.backward(loss2)

            # More efficient way to handle multiple losses:
            # The fp16->fp32 grad copy is delayed until fp16 grads from all
            # losses have been accumulated.
            optimizer.backward(loss1, update_fp32_grads=False)
            optimizer.backward(loss2, update_fp32_grads=False)
            optimizer.update_fp32_grads()
        """
        self.loss_scaler.backward(loss.float())
        if update_fp32_grads:
            self.update_fp32_grads()

    def update_fp32_grads(self):
        """
        Copy the .grad attribute from stored references to fp16 parameters to
        the .grad attribute of the main fp32 parameters that are directly
        updated by the optimizer.  :attr:`update_fp32_grads` only needs to be called if
        fp16_optimizer_obj.backward was called with update_fp32_grads=False.
        """
        if self.dynamic_loss_scale:
            self._check_overflow()
            if self.overflow: return
        self._copy_grads_fp16_to_fp32()
        self._downscale_fp32()

    def inspect_fp32_grad_data(self):
        """
        When running with FP16_Optimizer, .grad attributes of a model's fp16 leaves should not be
        regarded as truthful, because they might be scaled.
        After a call to :attr:`fp16_optimizer_obj.backward(loss)`, if no overflow was encountered,
        the fp32 main params' .grad
        attributes will contain valid gradients properly divided by the loss scale.  However,
        because :attr:`FP16_Optimizer` flattens some parameters, accessing them may be
        nonintuitive.  :attr:`inspect_fp32_grad_data`
        allows those gradients to be viewed with shapes corresponding to their associated model leaves.

        Returns:
            List of lists (one list for each parameter group).  The list for each parameter group
            is a list of the .grad.data attributes of the fp32 main params belonging to that group.
        """
        raise NotImplementedError("Currently not implemented, working on it...")
        fp32_grads_each_group = []
        if self.overflow:
            print("Warning:  calling FP16_Optimizer.inspect_fp32_grad_data while in an overflow state.  "
                  "Gradients are currently invalid (may be inf, nan, or stale).  Returning None.")
            return None
        else:
            return None


    @property
    def loss_scale(self):
        return self.loss_scaler.loss_scale


