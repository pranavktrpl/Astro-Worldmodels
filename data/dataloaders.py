import torch
import numpy as np
# from torch.utils.data.distributed import DistributedSampler
from .galaxies_source import GalaxiesSource
from .desiSpectra_source import DesiSpectraSource
from .SpectraTransforms import AstroSpectraMultiCropTransform

class MyDataset(torch.utils.data.IterableDataset):#, DistributedSampler):
    def __init__(self, split = "train", dataset = "Smith42/galaxies", columns = ["image", "image_crop", "galaxy_size"], shuffle = True, world_size = 1, rank = 0, Vg = 2, Vl = 8):
        self.ds = GalaxiesSource(dataset, columns, split)
        self.world_size = world_size
        self.rank = rank
        self.shuffle = shuffle
        self.epoch = 0
        self.stream = None
        
        # Spectra-specific training imports dataloaders too, so keep image augmentation deps lazy for image-only use.
        from .AstroTransforms import AstroMultiCropTransform
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


class DesiSpectraDataset(torch.utils.data.IterableDataset):#, DistributedSampler):
    def __init__(
        self,
        split = "train",
        dataset = "MultimodalUniverse/desi",
        columns = ["spectrum"],
        shuffle = True,
        world_size = 1,
        rank = 0,
        Vg = 2,
        Vl = 8,
        patch_size = 20,
        pad_value = 0.0,
        global_scale = (0.947, 1.0),
        local_scale = (0.20, 0.394),
        normalize = "none",
        view_mask_ratio = 0.0,
    ):
        self.ds = DesiSpectraSource(dataset, columns, split)
        self.world_size = world_size
        self.rank = rank
        self.shuffle = shuffle
        self.epoch = 0
        self.stream = None
        self.transformer = AstroSpectraMultiCropTransform(
            Vl,
            Vg,
            global_scale=global_scale,
            local_scale=local_scale,
            patch_size=patch_size,
            pad_value=pad_value,
            normalize=normalize,
            view_mask_ratio=view_mask_ratio,
        )

    def set_epoch(self, epoch):
        self.epoch = epoch

    def _spectrum_to_array(self, spectrum):
        return np.asarray(spectrum["flux"], dtype=np.float32)[None, :]

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

        for sample in shard_dataset:
            spectrum = self._spectrum_to_array(sample["spectrum"])
            yield self.transformer.augment_spectrum(spectrum)
