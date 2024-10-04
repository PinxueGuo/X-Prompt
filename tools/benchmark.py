# Modified from 
# https://github.com/hkchengrex/vos-benchmark
from vos_benchmark.benchmark import benchmark
from argparse import ArgumentParser


def evaluate(args):
    benchmark([args.g], [args.r])


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('-g', type=str)
    parser.add_argument('-r', type=str)
    args = parser.parse_args()
    evaluate(args)