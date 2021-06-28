import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import *
def pgd_attack(model, x, y, step_size, epsilon, perturb_steps,
                random_start=None, distance='l_inf'):
    model.eval()
    batch_size = len(x)
    if random_start:
        x_adv = x.detach() + random_start * torch.randn(x.shape).cuda().detach()
    else:
        x_adv = x.detach()
    if distance == 'l_inf':
        for i in range(perturb_steps):
            x_adv.requires_grad_()
            with torch.enable_grad():
                loss_c = F.cross_entropy(model(x_adv), y)
            grad = torch.autograd.grad(loss_c, [x_adv])[0]
            x_adv = x_adv.detach() + step_size * torch.sign(grad.detach())
            x_adv = torch.min(torch.max(x_adv, x - epsilon), x + epsilon)
            x_adv = torch.clamp(x_adv, 0.0, 1.0)
    return x_adv

class PGDAttack():
    def __init__(self, step_size, epsilon, perturb_steps,
                random_start=None):
        self.step_size = step_size
        self.epsilon = epsilon
        self.perturb_steps = perturb_steps
        self.random_start = random_start

    def __call__(self, model, x, y):
        model.eval()
        v=0
        m=1
        if self.random_start:
            x_adv = x.detach() + self.random_start * torch.randn(x.shape).cuda().detach()
        else:
            x_adv = x.detach()
        for i in range(self.perturb_steps):
            x_adv.requires_grad_()
            with torch.enable_grad():
                 loss_c = F.cross_entropy(model(x_adv), y)
            grad = torch.autograd.grad(loss_c, [x_adv])[0]
            v=m*v+self.step_size * torch.sign(grad.detach())
            pertubed = torch.sign(torch.randn_like(x_adv))
            x_adv = x_adv.detach() + v
            x_adv = torch.min(torch.max(x_adv, x - self.epsilon), x + self.epsilon)
            x_adv = torch.clamp(x_adv, 0.0, 1.0)
        return x_adv    

class FSGM_Attack():
    def __init__(self, step_size, epsilon,
                random_start=None):
        self.step_size = step_size
        self.epsilon = epsilon
        self.random_start = random_start

    def __call__(self, model, x, y):
        model.eval()
        if self.random_start:
            x_adv = x.detach() + self.random_start * torch.randn(x.shape).cuda().detach()
        else:
            x_adv = x.detach()
        x_adv.requires_grad_()
        with torch.enable_grad():
            loss_c = F.cross_entropy(model(x_adv), y)
        grad = torch.autograd.grad(loss_c, [x_adv])[0]
        sign_data_grad = grad.sign()
        perturbed_image = x + self.epsilon*sign_data_grad
        perturbed_image = torch.clamp(perturbed_image, 0, 1)
        return perturbed_image
def fsgm_attack(model, x, y, step_size, epsilon,
                random_start=None, distance='l_inf'):
    model.eval()
    if random_start:
        x_adv = x.detach() + random_start * torch.randn(x.shape).cuda().detach()
    else:
        x_adv = x.detach()
    x_adv.requires_grad_()
    with torch.enable_grad():
        loss_c = F.cross_entropy(model(x_adv), y)
    grad = torch.autograd.grad(loss_c, [x_adv])[0]
    sign_data_grad = grad.sign()
    perturbed_image = x + epsilon*sign_data_grad
    perturbed_image = torch.clamp(perturbed_image, 0, 1)
    return perturbed_image
class P(nn.Module):
    def __init__(self, x_adv):
        super().__init__()
        self.params = nn.Parameter(x_adv)


class MyAttack():
    def __init__(self, step_size, epsilon, perturb_steps,forward,backword,
                 random_start=None):
        self.step_size = step_size
        self.epsilon = epsilon
        self.perturb_steps = perturb_steps
        self.random_start = random_start

    def __call__(self, model, x, y, forward,backword):
        model.eval()
        v=0
        m=1
        if self.random_start:
            x_adv = x.detach() + self.random_start * torch.randn(x.shape).cuda().detach()
        else:
            x_adv = x.detach()
        for i in range(self.perturb_steps):
            x_adv.requires_grad_()
            with torch.enable_grad():
                 loss_c = F.cross_entropy(model(x_adv), y)
                 forward=forward+1
            grad = torch.autograd.grad(loss_c, [x_adv])[0]
            backword=backword+1
            v=m*v+self.step_size * torch.sign(grad.detach())
            x_adv = x_adv.detach() + v
            x_adv = torch.min(torch.max(x_adv, x - self.epsilon), x + self.epsilon)
            x_adv = torch.clamp(x_adv, 0.0, 1.0)
        index = list(range(len(x_adv)))
        x= x.cuda().detach().clone()
        for i in range(self.perturb_steps):
            if len(index) == 0:
                break
            data = x_adv[index]
            pred = model(data).cuda()
            forward=forward+1
            al_grad = torch.autograd.functional.jacobian(
                lambda cx: model(cx).sum(0), data
            ).transpose(0, 1).cuda()
            backword=backword+1
            for i, j in enumerate(index):
                sample = data[i].cuda()
                num_classes = pred.shape[-1]
                noise = float('inf')
                label = y[j].item()
                grad2 = al_grad[i, label]
                backword=backword+1
                min1 = torch.zeros_like(grad2)
                backword=backword+1
                for k in range(num_classes):
                    if k == label:
                        continue
                    gd1 = al_grad[i, k]
                    temp = gd1 - grad2
                    f_k = pred[i, k] - pred[i, label]

                    asd = abs(f_k.item()) / \
                        torch.linalg.norm(temp.flatten(), 1)
                    if asd < noise:
                        noise = asd
                        min1 = temp

                noise = noise * min1 / \
                    torch.linalg.norm(min1.flatten(), 1)
                pertubed_sample = sample + self.step_size * torch.sign(noise)
                pertubed_sample = torch.min(
                    torch.max(pertubed_sample, x[j] - self.epsilon), x[j] + self.epsilon)
                pertubed_sample = torch.clamp(pertubed_sample, 0.0, 1.0)

                x_adv[j] = pertubed_sample.to(x_adv.device)
            prediction = model(x_adv[index])
            forward=forward+1
            predicted_labels = prediction.argmax(-1)
            index = [
                idx for idx, a, b in zip( index, predicted_labels, y[index])
                if a == b
            ]

        return x_adv
def my_attack(model, x, y, step_size, epsilon, perturb_steps,forword,backword,
                random_start=None, distance='l_inf'):
    model.eval()
    v=0
    m=1
    if random_start:
        x_adv = x.detach() + random_start * torch.randn(x.shape).cuda().detach()
    else:
        x_adv = x.detach()
    for i in range(perturb_steps):
        x_adv.requires_grad_()
        with torch.enable_grad():
             loss_c = F.cross_entropy(model(x_adv), y)
             forword=forword+1
        grad = torch.autograd.grad(loss_c, [x_adv])[0]
        backword=backword+1
        v=m*v+step_size * torch.sign(grad.detach())
        x_adv = x_adv.detach() + v
        x_adv = torch.min(torch.max(x_adv, x - epsilon), x + epsilon)
        x_adv = torch.clamp(x_adv, 0.0, 1.0)
    index = list(range(len(x_adv)))
    x= x.cuda().detach().clone()
    for i in range(perturb_steps):
        if len(index) == 0:
            break
        data = x_adv[index]
        pred = model(data).cuda()
        forword=forword+1
        all_gradient = torch.autograd.functional.jacobian(
            lambda cx: model(cx).sum(0), data
        ).transpose(0, 1).cuda()
        backword=backword+1
        for i, j in enumerate(index):
            sample = data[i].cuda()
            num_classes = pred.shape[-1]
            noise = float('inf')
            label = y[j].item()
            original_gradient = all_gradient[i, label]
            backword=backword+1
            w_argmin = torch.zeros_like(original_gradient)
            for k in range(num_classes):
                if k == label:
                    continue
                current_gradient = all_gradient[i, k]
                backword=backword+1
                w_k = current_gradient - original_gradient
                f_k = pred[i, k] - pred[i, label]

                candidate = abs(f_k.item()) / \
                    torch.linalg.norm(w_k.flatten(), 1)
                if candidate < noise:
                    noise = candidate
                    w_argmin = w_k

            noise = noise * w_argmin / \
                torch.linalg.norm(w_argmin.flatten(), 1)
            pertubed_sample = sample + step_size * torch.sign(noise)
            pertubed_sample = torch.min(
                torch.max(pertubed_sample, x[j] - epsilon), x[j] + epsilon)
            pertubed_sample = torch.clamp(pertubed_sample, 0.0, 1.0)

            x_adv[j] = pertubed_sample.to(x_adv.device)
        prediction = model(x_adv[index])
        forword=forword+1
        predicted_labels = prediction.argmax(-1)
        index = [
            idx for idx, a, b in zip( index, predicted_labels, y[index])
            if a == b
        ]

    return x_adv
class random_attack():
    def __init__(self, step_size, epsilon, perturb_steps,
                 random_start=None):
        self.step_size = step_size
        self.epsilon = epsilon
        self.perturb_steps = perturb_steps * 10
        self.random_start = random_start
    def __call__(self, model, x, y):
        model.eval()
        x_adv = x.detach()
        loss1 = F.cross_entropy(model(x_adv), y)
        for i in range(self.perturb_steps):
            pertubed = torch.sign(torch.randn_like(x_adv))
            x_adv = x_adv.detach() + pertubed.detach()
            x_adv=x_adv = torch.min(
                torch.max(x_adv, x - self.epsilon), x + self.epsilon)
            loss2= F.cross_entropy(model(x_adv), y)
            x_adv = torch.clamp(x_adv, 0.0, 1.0)
        return x_adv

