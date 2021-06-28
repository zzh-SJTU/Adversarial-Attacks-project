# -*- coding: utf-8 -*-
"""
Created on Sun Jun 13 20:33:09 2021

@author: zzh
"""
from eval_model import eval_model
from utils import get_test_cifar
import torch
import torch.nn as nn
import argparse
from models import ResNet18

def parse_infer_args():
    parser = argparse.ArgumentParser(
        description=' without Attack')
    parser.add_argument(
        '--batch_size', type=int, default=128, metavar='N',)
    parser.add_argument(
        '--model_path', type=str,
        default=''
    )
    parser.add_argument('--device', type=str, default="cuda:0")

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_infer_args()
    device = torch.device(args.device)

    model = ResNet18().to(device)
    model.load_state_dict(torch.load('res18_e99_0.8094_0.5521-final.pt'))
    model = nn.DataParallel(model, device_ids=[0])

    model.eval()
    test_loader = get_test_cifar(args.batch_size)
    correct = []
    num = 0
    natural_acc, _ = eval_model(model, test_loader, args.device)
    print(
        f"Natual Acc: {natural_acc:.5f}"
    )
