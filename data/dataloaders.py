import torch
from torch.utils.data import DataLoader
# from torch.utils.data.distributed import DistributedSampler
from .galaxies_source import GalaxiesSource

class HFDataset(torch.utils.data.IterableDataset):#, DistributedSampler):
    def __init__(self, split, dataset = "Smith42/galaxies", columns = ["image", "image_crop", "galaxy_size"], shuffle = True, world_size = 1, rank = 0):
        self.ds = GalaxiesSource(dataset, columns, split)
        self.world_size = world_size
        self.rank = rank
        self.shuffle = shuffle
        self.epoch = 0

    def __iter__(self):
        if self.shuffle:
            stream = self.ds.load_dataset().shuffle(seed=42, buffer_size=50_000).set_epoch(self.epoch)
        else:
            stream = self.ds.load_dataset().set_epoch(self.epoch)
        self.worker_info = torch.utils.data.get_worker_info()
        num_workers = self.worker_info.num_workers if self.worker_info else 1
        worker_id = self.worker_info.id if self.worker_info else 0
        num_shards = num_workers * self.world_size
        shard_id = self.rank * num_workers + worker_id
        shard_dataset = stream.shard(num_shards=num_shards, index=shard_id)
        return iter(shard_dataset)
