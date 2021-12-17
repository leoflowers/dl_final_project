import utils.data as data
import utils.stats as stats
import model.model as m
import model.training as training
import model.testing as testing

import os
import torch
import argparse

import torch.distributed as dist
import torch.multiprocessing as mp

from torch.nn.parallel import DistributedDataParallel


def init_process(rank, size, backend='gloo'):
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29500'
    dist.init_process_group(backend, rank=rank, world_size=size)


def modelTraining(epochs, distributed=False, world_size=1, rank=0):
    #if device is None:
    device = torch.device(f'cuda:{rank}' if torch.cuda.is_available() else 'cpu')
    print(f"Training on {device}")

    if distributed:
        init_process(rank, world_size)
        torch.cuda.set_device(rank)

    torch.cuda.manual_seed_all(1337)

    # load data
    datasets = data.create_datasets(data.data_transforms(), data.check_valid)
    dataloaders = data.create_dataloaders(datasets, distributed, world_size, rank)


    # initialize model
    model = m.ResNet34().to(device)
    criterion = torch.nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.Adam(model.model.parameters(), lr=1e-3)

    if distributed:
        model = DistributedDataParallel(model, device_ids=[rank], output_device=rank)


    # training/eval loop
    epochs = epochs
    loss_per_epoch = []
    acc_per_epoch = []
    test_loss_per_epoch = []

    for curr_epoch in range(epochs):
        print(f"---------------------- epoch {curr_epoch + 1}/{epochs} ----------------------")

        # if distributed, tell sampler the epoch
        if distributed:
            dataloaders['train'].sampler.set_epoch(curr_epoch)

        # trains first
        model.model.train()
        curr_epoch_loss = training.train(model, dataloaders['train'], criterion, optimizer, device)
        loss_per_epoch.append(curr_epoch_loss)
        print(f"\taverage loss: {loss_per_epoch[curr_epoch]}")
        
        # then tests model
        model.model.eval()
        curr_epoch_acc, curr_epoch_test_loss = testing.test(model, dataloaders['test'], criterion, device)
        acc_per_epoch.append(curr_epoch_acc)
        test_loss_per_epoch.append(curr_epoch_test_loss)
        print(f"\taccuracy: {acc_per_epoch[curr_epoch]}")


    # saves metrics into csv
    filename = f"../stats/resnet34-{world_size}gpus.csv"
    stats.collectStatistics(loss_per_epoch, acc_per_epoch, test_loss_per_epoch, filename)


    # saves model
    if rank == 0:
        torch.save(model.model.state_dict(), f'../model_states/resnet34-{world_size}gpus-{epochs}epochs.pt')

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--world_size", "-w", type=int, default=1, help="# of GPUs to use")
    parser.add_argument("--rank", "-r", type=int, default=0, help="rank of process")
    parser.add_argument("--epochs", "-e", type=int, default=100, help="number of epochs to train model for")
    args = parser.parse_args()
  
    #if args.world_size==1:
        # trains model locally with whatever device is available
    distributed = (args.world_size > 1)
    modelTraining(args.epochs, distributed, args.world_size, args.rank)
"""
    else:
        # trains model on multiple devices
        size = args.processes 
        processes = []
        mp.set_start_method("spawn")
        for rank in range(size):
            p = mp.Process(target=init_process, args=(rank, size, run))
            p.start()
            processes.append(p)

        for p in processes:
            p.join()
"""
