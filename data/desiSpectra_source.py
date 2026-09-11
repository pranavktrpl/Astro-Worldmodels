from datasets import load_dataset
from pathlib import Path
import pyarrow.parquet as pq


class DesiSpectraSource():
    def __init__(
        self,
        dataset="MultimodalUniverse/desi",
        columns=[
            "spectrum",
            "Z",
            "ZERR",
            "EBV",
            "FLUX_G",
            "FLUX_R",
            "FLUX_Z",
            "FLUX_IVAR_G",
            "FLUX_IVAR_R",
            "FLUX_IVAR_Z",
            "FIBERFLUX_G",
            "FIBERFLUX_R",
            "FIBERFLUX_Z",
            "FIBERTOTFLUX_G",
            "FIBERTOTFLUX_R",
            "FIBERTOTFLUX_Z",
            "ZWARN",
            "object_id",
        ],
        split="train",
    ):
        self.dataset = dataset
        self.columns = columns
        self.split = split
        dataset_path = Path(dataset)
        self.parquet_files = (
            sorted(dataset_path.rglob("*.parquet")) if dataset_path.exists() else None
        )

    def balanced_file_shards(self, num_shards):
        if self.parquet_files is None:
            return None
        if num_shards < 1:
            raise ValueError("num_shards must be positive")

        files_with_rows = [
            (path, pq.ParquetFile(path).metadata.num_rows)
            for path in self.parquet_files
        ]
        assignments = [[] for _ in range(num_shards)]
        assigned_rows = [0 for _ in range(num_shards)]
        for path, rows in sorted(files_with_rows, key=lambda item: item[1], reverse=True):
            shard_id = min(range(num_shards), key=assigned_rows.__getitem__)
            assignments[shard_id].append(str(path))
            assigned_rows[shard_id] += rows
        return assignments

    def load_dataset(self, data_files=None):
        if self.parquet_files is not None:
            parquet_files = data_files or [str(path) for path in self.parquet_files]
            if not parquet_files:
                raise FileNotFoundError(f"No Parquet files found under {self.dataset}")
            return load_dataset(
                "parquet",
                data_files={"train": parquet_files},
                columns=self.columns,
                split=self.split,
                streaming=True,
            )

        dataset = load_dataset(self.dataset, columns=self.columns, split=self.split, streaming=True)
        return dataset

        ## Sharding is done at dataloader level too, not at the dataset level, cause no of shards != No of GPUs/worlds, but no of #shards = #worlds * #processes. Cause when you do DataLoader(dataset, num_workers = 2), it creats 2 worker subprocesses under each GPU process.
        #sharding to split across multiple processes/GPUs
        # if world_size > 1:
        #     dataset = dataset.shard(num_shards=world_size, index=rank)

        ## Batching should be done at dataloader level, not at the dataset level
        #Not the actual batch size, but the mini_batch size/no of worlds ==> Equal split across all processes/GPUs
        # if batch_size is not None:
        #     batched_dataset = dataset.batch(batch_size=batch_size)
        #     return iter(batched_dataset)
