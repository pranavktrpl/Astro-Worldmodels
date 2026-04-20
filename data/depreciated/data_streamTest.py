import argparse
import time
from statistics import mean

import torch
from torch.utils.data import DataLoader

from data.dataloaders import MyDataset

from dotenv import load_dotenv

load_dotenv()

def sync_cuda():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def benchmark_dataset_iter(dataset, warmup=10, steps=100):
    print("\n" + "=" * 80)
    print("Benchmark 1: dataset iterator (sample-level)")
    print("=" * 80)

    it = iter(dataset)

    # Warmup
    for _ in range(warmup):
        _ = next(it)

    times = []
    for i in range(steps):
        t0 = time.perf_counter()
        sample = next(it)
        sync_cuda()
        t1 = time.perf_counter()
        times.append(t1 - t0)

        if i == 0:
            print("First sample keys:", list(sample.keys()))
            for k, v in sample.items():
                if torch.is_tensor(v):
                    print(f"  {k}: shape={tuple(v.shape)}, dtype={v.dtype}")
                else:
                    print(f"  {k}: type={type(v)}")

    avg = mean(times)
    print(f"Samples tested        : {steps}")
    print(f"Avg sec / sample      : {avg:.6f}")
    print(f"Samples / sec         : {1.0 / avg:.3f}")
    print(f"Min sec / sample      : {min(times):.6f}")
    print(f"Max sec / sample      : {max(times):.6f}")

    return avg


def benchmark_dataloader(loader, warmup=5, steps=50, batch_size=1, total_views=1):
    print("\n" + "=" * 80)
    print("Benchmark 2: dataloader (batch-level)")
    print("=" * 80)

    it = iter(loader)

    # Warmup
    for _ in range(warmup):
        _ = next(it)

    times = []
    for i in range(steps):
        t0 = time.perf_counter()
        batch = next(it)
        sync_cuda()
        t1 = time.perf_counter()
        times.append(t1 - t0)

        if i == 0:
            print("First batch keys:", list(batch.keys()) if isinstance(batch, dict) else type(batch))
            if isinstance(batch, dict):
                for k, v in batch.items():
                    if torch.is_tensor(v):
                        print(f"  {k}: shape={tuple(v.shape)}, dtype={v.dtype}")
                    else:
                        print(f"  {k}: type={type(v)}")

    avg = mean(times)
    base_imgs_per_sec = batch_size / avg
    view_imgs_per_sec = (batch_size * total_views) / avg

    print(f"Batches tested        : {steps}")
    print(f"Avg sec / batch       : {avg:.6f}")
    print(f"Batches / sec         : {1.0 / avg:.3f}")
    print(f"Base imgs / sec       : {base_imgs_per_sec:.3f}")
    print(f"View imgs / sec       : {view_imgs_per_sec:.3f}")
    print(f"Min sec / batch       : {min(times):.6f}")
    print(f"Max sec / batch       : {max(times):.6f}")

    return avg


def estimate_epoch_time(train_images, imgs_per_sec):
    total_sec = train_images / imgs_per_sec
    hours = total_sec / 3600
    print("\n" + "=" * 80)
    print("Estimated epoch time")
    print("=" * 80)
    print(f"Train images          : {train_images:,}")
    print(f"Base imgs / sec       : {imgs_per_sec:.3f}")
    print(f"Estimated epoch secs  : {total_sec:.1f}")
    print(f"Estimated epoch hours : {hours:.2f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--dataset", type=str, default="Smith42/galaxies")
    parser.add_argument("--columns", nargs="+", default=["image_crop", "galaxy_size"])
    parser.add_argument("--shuffle", action="store_true")
    parser.add_argument("--world-size", type=int, default=1)
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--Vg", type=int, default=2)
    parser.add_argument("--Vl", type=int, default=0)

    parser.add_argument("--batch-size", type=int, default=96)
    parser.add_argument("--num-workers", type=int, default=16)

    parser.add_argument("--sample-warmup", type=int, default=10)
    parser.add_argument("--sample-steps", type=int, default=50)

    parser.add_argument("--batch-warmup", type=int, default=5)
    parser.add_argument("--batch-steps", type=int, default=20)

    parser.add_argument("--train-images", type=int, default=8330000)

    args = parser.parse_args()

    print("\nCreating dataset...")
    dataset = MyDataset(
        split=args.split,
        dataset=args.dataset,
        columns=args.columns,
        shuffle=args.shuffle,
        world_size=args.world_size,
        rank=args.rank,
        Vg=args.Vg,
        Vl=args.Vl,
    )
    dataset.set_epoch(0)

    total_views = args.Vg + args.Vl

    # Sample-level benchmark
    sample_avg = benchmark_dataset_iter(
        dataset,
        warmup=args.sample_warmup,
        steps=args.sample_steps,
    )

    # Recreate dataset for dataloader benchmark so iterator state is fresh
    dataset = MyDataset(
        split=args.split,
        dataset=args.dataset,
        columns=args.columns,
        shuffle=args.shuffle,
        world_size=args.world_size,
        rank=args.rank,
        Vg=args.Vg,
        Vl=args.Vl,
    )
    dataset.set_epoch(0)

    print("\nCreating dataloader...")
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    batch_avg = benchmark_dataloader(
        loader,
        warmup=args.batch_warmup,
        steps=args.batch_steps,
        batch_size=args.batch_size,
        total_views=total_views,
    )

    base_imgs_per_sec = args.batch_size / batch_avg
    estimate_epoch_time(args.train_images, base_imgs_per_sec)


if __name__ == "__main__":
    main()