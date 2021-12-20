# dl_final_project
To run, first change into src/ directory and then run the following command:
 
  `python -m torch.distributed.launch --nproc_per_node=(number of gpus wanted) main.py --local_world_size=(same as --nproc_per_node)`
  
So if 2 GPUs are available, run the script as

  `python -m torch.distributed.launch --nproc_per_node=2 main.py --local_world_size 2`
