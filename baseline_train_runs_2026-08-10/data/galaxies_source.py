from PIL import Image
from datasets import load_dataset

class GalaxiesSource():
    def __init__(self, dataset="Smith42/galaxies", columns=["image", "image_crop", "galaxy_size"], split="train"):
        self.dataset = dataset
        self.columns = columns
        self.split = split

    def load_dataset(self):
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
