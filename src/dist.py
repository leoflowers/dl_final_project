import utils.data as data
import model.model as m
import model.training as training
import model.testing as testing

#import pytorch_lightning as pl
import os
import torch
import argparse

import torch.distributed as dist
import torch.multiprocessing as mp

from torch.nn.parallel import DistributedDataParallel as DPP


def run(rank, size):
    group = dist.new_group([0,1,2,3])
    tensor = torch.ones(1)
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM, group=group)
    print(f"Rank {rank} has data {tensor[0]}")


def setup(rank, world_size, backend='gloo'):
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29500'
    
    dist.init_process_group(backend, rank=rank, world_size=world_size)
    #func(rank, size)


def cleanup():
    dist.destroy_process_group()


def main(rank, world_size):
    print(f"Training model on rank {rank}")
    setup(rank, world_size)

    model = m.ResNet34()
    dpp_model = DPP(model, device_ids=[rank])    

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(dpp_model.parameters(), lr=1e-3)

    epochs = 5
    training.train(model, dataloaders['train'], criterion, optimizer, epochs, rank)

    cleanup()
    #testing.test(model, dataloaders['test'], device)
    #trainer.fit(model, dataloaders['train'], dataloaders['test'])


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--processes", "-p", type=int, help="number of processes desired for training")
    args = parser.parse_args()
        
    #main()

    world_size = args.processes 
    processes = []
    mp.set_start_method("spawn")
    for rank in range(size):
        p = mp.Process(target=main, args=(rank, world_size))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

