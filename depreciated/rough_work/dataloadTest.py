import torch
import torch.distributed as dist
from torch.utils.data import DataLoader

from data.dataloaders import MyDataset

dist.init_process_group(backend="nccl")

world_size = dist.get_world_size()
rank = dist.get_rank()

print(f"world_size = {world_size}, rank = {rank}")
dataset = MyDataset("train", "Smith42/galaxies", ["image", "image_crop", "galaxy_size"], True, world_size, rank)

loader = DataLoader(dataset, batch_size = 4, num_workers = 2)

epochs = 8

for epoch in range(epochs):
    dataset.set_epoch(epoch)
    counter = 0
    for batch in loader:
        print("#"*15)
        print(f"epoch = {epoch}, batch = {counter}, rank = {rank}/{world_size}")
        print("batch length = ", len(batch))
        print("batch keys = ", batch.keys())
        print("galaxy size = ", batch["galaxy_size"])
        counter += 1
        if counter > 10:
            break
    dist.barrier()