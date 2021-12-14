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


def run(rank, size):
    group = dist.new_group([0,1,2,3])
    tensor = torch.ones(1)
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM, group=group)
    print(f"Rank {rank} has data {tensor[0]}")


def init_process(rank, size, func, backend='gloo'):
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29500'
    dist.init_process_group(backend, rank=rank, world_size=size)
    func(rank, size)


def modelTraining():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    datasets = data.create_datasets(data.data_transforms(), data.check_valid)
    dataloaders = data.create_dataloaders(datasets)

    criterion = torch.nn.CrossEntropyLoss()
    model = m.Model(criterion)
    model.move_to_device(device)
    optimizer = torch.optim.Adam(model.model.parameters(), lr=1e-3)

    epochs = 5
    training.train(model, dataloaders['train'], optimizer, epochs, device)
    #testing.test(model, dataloaders['test'], device)
    #trainer.fit(model, dataloaders['train'], dataloaders['test'])


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--processes", "-p", type=int, help="number of processes desired for training")
    args = parser.parse_args()
        
    main()

    size = args.processes 
    processes = []
    mp.set_start_method("spawn")
    for rank in range(size):
        p = mp.Process(target=init_process, args=(rank, size, run))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

