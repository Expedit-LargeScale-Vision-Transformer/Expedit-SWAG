import argparse
import hubconf
import time
from typing import List, Tuple, Union

import torch
from tqdm import tqdm

from tools.flop_count import flop_count
from tools.parser import add_model_args, get_model_args


def get_parser():
    model_names = [
        name
        for name in hubconf.__dict__
        if "in1k" in name and callable(hubconf.__dict__[name])
    ]

    parser = argparse.ArgumentParser(description="PyTorch ImageNet Training")
    parser.add_argument(
        "-m",
        "--model",
        metavar="MODEL",
        default="vit_b16_in1k",
        choices=model_names,
        help="model name: " + " | ".join(model_names) + " (default: vit_b16_in1k)",
    )
    parser.add_argument(
        "-r", "--resolution", default=224, type=int, help="input resolution of the images"
    )
    parser.add_argument(
        "-b",
        "--batch-size",
        default=200,
        type=int,
        metavar="N",
        help="mini-batch size (default: 200), this is the total "
             "batch size of all GPUs on the current node when "
             "using Data Parallel or Distributed Data Parallel",
    )
    add_model_args(parser)

    return parser


def benchmark(
    model: torch.nn.Module,
    device: torch.device = 0,
    input_size: Tuple[int] = (3, 224, 224),
    batch_size: int = 64,
    runs: int = 40,
    throw_out: float = 0.25,
    use_fp16: bool = False,
    verbose: bool = False,
) -> float:
    """
    Benchmark the given model with random inputs at the given batch size.

    Args:
     - model: the module to benchmark
     - device: the device to use for benchmarking
     - input_size: the input size to pass to the model (channels, h, w)
     - batch_size: the batch size to use for evaluation
     - runs: the number of total runs to do
     - throw_out: the percentage of runs to throw out at the start of testing
     - use_fp16: whether or not to benchmark with float16 and autocast
     - verbose: whether or not to use tqdm to print progress / print throughput at end

    Returns:
     - the throughput measured in images / second
    """
    if not isinstance(device, torch.device):
        device = torch.device(device)
    is_cuda = torch.device(device).type == "cuda"

    model = model.eval().to(device)
    input = torch.rand(batch_size, *input_size, device=device)
    if use_fp16:
        input = input.half()

    warm_up = int(runs * throw_out)
    total = 0
    start = time.time()

    with torch.autocast(device.type, enabled=use_fp16):
        with torch.no_grad():
            for i in tqdm(range(runs), disable=not verbose, desc="Benchmarking"):
                if i == warm_up:
                    if is_cuda:
                        torch.cuda.synchronize()
                    total = 0
                    start = time.time()

                model(input)
                total += batch_size

    if is_cuda:
        torch.cuda.synchronize()

    end = time.time()
    elapsed = end - start

    throughput = total / elapsed

    if verbose:
        print(f"Throughput: {throughput:.2f} im/s")

    return throughput

def main():
    parser = get_parser()
    args = parser.parse_args()
    print(args)

    model_args = get_model_args(args)
    # create model
    model = torch.hub.load("./", args.model, source="local", image_size=args.resolution,
        **model_args
    )

    device = "cuda:0"
    runs = 50
    input_size = (3, args.resolution, args.resolution)

    throughput = benchmark(
        model,
        device=device,
        input_size=(3, args.resolution, args.resolution),
        batch_size=args.batch_size,
        runs=runs,
        verbose=True,
    )

    sample = torch.rand(1, *input_size, device=device)
    flops = flop_count(model, (sample, ))
    print("Flops: {:.5g} G".format(sum(flops.values())))


if __name__ == "__main__":
    main()
