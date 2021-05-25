import torch
import torch.nn as nn

class ReLUAlphaFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input, alpha):
        ctx.save_for_backward(input)
        ctx.alpha = alpha
        return input.clamp(min=0)

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad_input[input < 0] = 0
        grad_input[input == 0] = ctx.alpha
        return grad_input, None


class ReLUAlpha(nn.Module):

    def __init__(self, alpha):
        super(ReLUAlpha, self).__init__()
        self.alpha = alpha

    def forward(self, input):
        return ReLUAlphaFunction.apply(input, self.alpha)


class ReLURandomFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return input.clamp(min=0)

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        rdm = torch.rand_like(grad_output)
        grad_input[input < 0] = 0
        grad_input[input == 0] = rdm[input == 0]
        return grad_input


relu_random = ReLURandomFunction.apply

class ReLURandom(nn.Module):

    def __init__(self):
        super(ReLURandom, self).__init__()

    def forward(self, input):
        return relu_random(input)

