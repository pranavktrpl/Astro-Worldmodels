import torch
import hashlib
# from torch.utils.data.distributed import DistributedSampler
from .galaxies_source import GalaxiesSource
from .desiSpectra_source import DesiSpectraSource
from .SpectraTransforms import AstroSpectraV2Transform


def spectra_split_for_object_id(object_id, seed=42):
    digest = hashlib.blake2b(
        f"{seed}:{object_id}".encode("utf-8"),
        digest_size=8,
        person=b"AstroJV2",
    ).digest()
    bucket = int.from_bytes(digest, byteorder="big") % 100
    if bucket < 98:
        return "train"
    if bucket == 98:
        return "validation"
    return "test"

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
        columns = ["spectrum", "object_id"],
        shuffle = True,
        world_size = 1,
        rank = 0,
        num_views = 2,
        patch_size = 20,
        expected_num_pixels = 7781,
        mask_ratio = 0.30,
        mask_span = (2, 12),
        disjoint_view_masks = True,
        min_valid_patch_fraction = 0.50,
        noise_scale = 0.50,
        max_normalized_noise_std = 3.0,
        split_seed = 42,
        shuffle_buffer_size = 50_000,
        loader_num_workers = 1,
    ):
        if split not in {"train", "validation", "test"}:
            raise ValueError(f"Unknown DESI logical split: {split}")
        self.ds = DesiSpectraSource(dataset, columns, "train")
        self.split = split
        self.split_seed = split_seed
        self.world_size = world_size
        self.rank = rank
        self.shuffle = shuffle
        self.shuffle_buffer_size = shuffle_buffer_size
        self.loader_num_workers = max(1, loader_num_workers)
        self.epoch = 0
        self.stream = None
        # Partition files once across DDP ranks. Hugging Face's IterableDataset
        # automatically partitions each rank's files across DataLoader workers.
        self.file_shards = self.ds.balanced_file_shards(
            self.world_size
        )
        self.transformer = AstroSpectraV2Transform(
            num_views=num_views,
            patch_size=patch_size,
            expected_num_pixels=expected_num_pixels,
            mask_ratio=mask_ratio,
            mask_span=mask_span,
            disjoint_view_masks=disjoint_view_masks,
            min_valid_patch_fraction=min_valid_patch_fraction,
            noise_scale=noise_scale,
            max_normalized_noise_std=max_normalized_noise_std,
        )

    def set_epoch(self, epoch):
        self.epoch = epoch

    def _belongs_to_split(self, sample):
        return spectra_split_for_object_id(sample["object_id"], self.split_seed) == self.split

    def __iter__(self):
        self.worker_info = torch.utils.data.get_worker_info()
        num_workers = self.worker_info.num_workers if self.worker_info else 1
        if num_workers != self.loader_num_workers:
            raise RuntimeError(
                f"Configured {self.loader_num_workers} loader workers, got {num_workers}"
            )
        data_files = (
            self.file_shards[self.rank] if self.file_shards is not None else None
        )

        self.stream = self.ds.load_dataset(data_files=data_files).filter(self._belongs_to_split)
        if self.shuffle:
            self.stream = self.stream.shuffle(
                seed=self.split_seed,
                buffer_size=self.shuffle_buffer_size,
            )
        self.stream.set_epoch(self.epoch)
        shard_dataset = (
            self.stream
            if self.file_shards is not None
            else self.stream.shard(num_shards=self.world_size, index=self.rank)
        )

        for sample in shard_dataset:
            transformed = self.transformer.augment_spectrum(sample["spectrum"])
            if transformed is None:
                continue
            transformed["object_id"] = sample["object_id"]
            yield transformed
