import torch
# from torch.utils.data.distributed import DistributedSampler
from .galaxies_source import GalaxiesSource
from .Astro-transforms import AstroMultiCropTransform

class MyDataset(torch.utils.data.IterableDataset):#, DistributedSampler):
    def __init__(self, split = "train", dataset = "Smith42/galaxies", columns = ["image", "image_crop", "galaxy_size"], shuffle = True, world_size = 1, rank = 0, Vg = 2, Vl = 8):
        self.ds = GalaxiesSource(dataset, columns, split)
        self.world_size = world_size
        self.rank = rank
        self.shuffle = shuffle
        self.epoch = 0
        self.stream = None
        
        self.transformer = AstroMultiCropTransform(Vl, Vg)
        
    def set_epoch(self, epoch):
        self.epoch = epoch

    def __iter__(self):
        if self.shuffle:
            self.stream = self.ds.load_dataset().shuffle(seed=42, buffer_size=50_000)
        else:
            self.stream = self.ds.load_dataset()

        self.stream.set_epoch(self.epoch)
        self.worker_info = torch.utils.data.get_worker_info()
        num_workers = self.worker_info.num_workers if self.worker_info else 1
        worker_id = self.worker_info.id if self.worker_info else 0
        num_shards = num_workers * self.world_size
        shard_id = self.rank * num_workers + worker_id
        # print(f"rank = {self.rank}, worker_info = {self.worker_info}, shard_id = {shard_id}, num_shards = {num_shards}")
        shard_dataset = self.stream.shard(num_shards=num_shards, index=shard_id)

        # transformer = TransformImage(1, 1)
        for sample in shard_dataset:
            # sample["image"] = transformer.augment_image(sample["image"])
            # sample["image_crop"] = self.transformer.augment_image(sample["image_crop"])
            # yield {"image_crop": self.transformer.augment_image(sample["image_crop"])}
            crops = self.transformer.augment_image(sample["image_crop"])
            yield crops

        # return iter(shard_dataset)
