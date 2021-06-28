# -*- coding: utf-8 -*-
"""
Created on Sun Jun 13 20:06:42 2021

@author: zzh
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision
from torchvision import datasets, transforms
from utils import prepare_cifar, get_test_cifar
from pgd_attack import FSGM_Attack
from pgd_attack import PGDAttack
from pgd_attack import MyAttack
from pgd_attack import random_attack
from models import  WideResNet, WideResNet34, WideResNet28
from models import ResNet18
from model import get_model_for_attack
from tqdm import tqdm, trange
from eval_model import eval_model, eval_model_pgd, eval_model_with_attack
import argparse
import os

def parse_args():
    parser = argparse.ArgumentParser(description='Test Robust Accuracy')
    parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                    help='input batch size for training (default: 128)')
    parser.add_argument('--step_size', type=int, default=0.003,
                    help='step size for pgd attack(default:0.003)')
    parser.add_argument('--epsilon', type=float, default=8/255.0,
                    help='max distance for pgd attack (default epsilon=8/255)')
    parser.add_argument('--perturb_steps', type=int, default=20,
                    help='iterations for pgd attack (default pgd20)')
    parser.add_argument('--model_name', type=str, default="")
    parser.add_argument('--model_path', type=str, default="./models/weights/model-wideres-pgdHE-wide10.pt")
    parser.add_argument('--gpu_id', type=str, default="0")
    return parser.parse_args()



if __name__=='__main__':
    forward=0
    backward=0
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id   #多卡机设置使用的gpu卡号
    gpu_num = max(len(args.gpu_id.split(',')), 1)
    device = torch.device('cuda')
    if args.model_name!="":
        model = get_model_for_attack(args.model_name).to(device)   # 根据model_name, 切换要攻击的model
        model = nn.DataParallel(model, device_ids=[i for i in range(gpu_num)])
        
    else:
        # 防御任务, Change to your model here
        model = ResNet18()
        model.load_state_dict(torch.load('res18_e99_0.8094_0.5521-final.pt'))
        model = nn.DataParallel(model, device_ids=[i for i in range(gpu_num)])
    #攻击任务：Change to your attack function here
    #下面几种攻击方法可以选择进行测试，去掉注释就行，主要是Myattack效果较好
    #attack = FSGM_Attack(args.step_size, args.epsilon)
    #attack = PGDAttack(args.step_size, args.epsilon, args.perturb_steps)
    attack = MyAttack(args.step_size, args.epsilon, args.perturb_steps,forward,backward)
    #attack = random_attack(args.step_size, args.epsilon, args.perturb_steps)
    model.eval()
    test_loader = get_test_cifar(args.batch_size)
    natural_acc, robust_acc, distance = eval_model_with_attack(model, test_loader, attack, device,forward,backward)
    print(f"Natural Acc: {natural_acc:.5f}, Robust acc: {robust_acc:.5f}, distance:{distance:.5f}")
