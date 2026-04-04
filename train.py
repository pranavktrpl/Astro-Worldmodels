import os
import random

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader

from data.dataloaders import MyDataset

######################################################### DDP Setup + Dataloading #########################################################
def setup_ddp():
    dist.init_process_group(backend="nccl")

    world_size = dist.get_world_size()
    rank = dist.get_rank()   #Which process this is globally, so rank is an init in the dataset object we created, cause every GPU gets a custom dataset object
    local_rank = int(os.environ["LOCAL_RANK"])  #Just which GPU this is on the node, same as rank, no need honestly

    torch.cuda.set_device(local_rank)  #Binds the process to the GPU, for if we stop mid way, and we want to resume, the process no is not mixed and we stick to same GPU always
    device = torch.device("cuda", local_rank)

    return world_size, rank, local_rank, device

def seed_everything(seed, rank):
    seed = seed + rank
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def is_main_process(rank):
    return rank == 0         #So we log only one main process, not all
    
def build_train_loader(cfg, world_size, rank):
    train_ds = MyDataset(
        split="train", dataset=cfg.dataset_name, columns=cfg.columns, shuffle=True, world_size=world_size, rank=rank,
    )
    train_loader = DataLoader(
        train_ds, batch_size=cfg.bs, num_workers=cfg.num_workers, pin_memory=True, persistent_workers=(cfg.num_workers > 0),
    )
    return train_ds, train_loader

def build_train_loader(cfg, world_size, rank):
    val_ds = MyDataset(
        split="validation", dataset=cfg.dataset_name, columns=cfg.columns, shuffle=False, world_size=1, rank=0,
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg.bs, num_workers=cfg.num_workers, pin_memory=True, persistent_workers=(cfg.num_workers > 0),
    )
    return val_ds, val_loader







def main():
    world_size, rank, local_rank, device = setup_ddp()
    seed_everything(seed=42, rank=rank)

    if is_main_process(rank):
        print(f"world_size = {world_size}, rank = {rank}, local_rank = {local_rank}")

    # temporary config placeholder
    class CFG:
        dataset_name = "Smith42/galaxies"
        columns = ["image", "image_crop", "galaxy_size"]
        bs = 4
        eval_bs = 8
        num_workers = 2

    cfg = CFG()

    train_ds, train_loader = build_train_loader(cfg, world_size, rank)
    val_ds, val_loader = build_val_loader(cfg, world_size, rank)

    for epoch in range(2):
        train_ds.set_epoch(epoch)

        for step, batch in enumerate(train_loader):
            if step > 3:
                break
            
            images = batch["image"].to(device, non_blocking=True)
            image_crops = batch["image_crop"].to(device, non_blocking=True)
            
            if is_main_process(rank):
                print("image device =", images.device)
                print("image_crop device =", image_crops.device)
                print("#" * 15)
                print(f"epoch = {epoch}, step = {step}")
                print("batch keys =", batch.keys())

                print("image shape =", batch["image"].shape)
                print("image_crop shape =", batch["image_crop"].shape)

                if "galaxy_size" in batch:
                    print("galaxy_size shape =", batch["galaxy_size"].shape)
                    print("galaxy_size =", batch["galaxy_size"])
        
        dist.barrier()
        
    dist.destroy_process_group()

if __name__ == "__main__":
    main()