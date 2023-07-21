import argparse
from typing import Optional

import numpy as np


def get_one_chunk(chunksize: int, chunk_num: int, base_type: str = "int") -> Optional[np.ndarray]:
    # returns an array representing a single chunk
    return np.ones((chunksize,), dtype=base_type)


def cast_last(chunks: int, chunksize: int, base_type: str = "int", cast_type: str = "float64"):
    chunk_list = []
    for i in range(chunks):
        chunk_list.append(get_one_chunk(chunksize, i, base_type=base_type))
    x = np.concatenate(chunk_list, axis=0).astype(cast_type)
    _ = x.mean()


def cast_first(chunks: int, chunksize: int, base_type: str = "int", cast_type: str = "float64"):
    chunk_list = []
    for i in range(chunks):
        chunk_vals = get_one_chunk(chunksize, i, base_type=base_type).astype(cast_type)
        chunk_list.append(chunk_vals)
    x = np.concatenate(chunk_list, axis=0)
    _ = x.mean()


def no_cast(chunks: int, chunksize: int, base_type: str = "int"): 
    chunk_list = []
    for i in range(chunks):
        chunk_vals = get_one_chunk(chunksize, i, base_type=base_type)
        chunk_list.append(chunk_vals)
    x = np.concatenate(chunk_list, axis=0)
    _ = x.mean() 


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('chunks', type=int, help='number of chunks (int)')
    parser.add_argument('chunksize', type=int, help='size of each chunk (int)')
    parser.add_argument('method', type=str, help="one of: cast_first, cast_last, no_cast")
    parser.add_argument('--base_type', type=str, help="the type of the base arrays", default='int')
    parser.add_argument('--cast_type', type=str, help="the type to cast the aggregated arrays to", default='float64')

    args = parser.parse_args()

    if args.method == "cast_first":
        cast_first(args.chunks, args.chunksize, base_type=args.base_type, cast_type=args.cast_type)
    elif args.method == "cast_last":
        cast_last(args.chunks, args.chunksize, base_type=args.base_type, cast_type=args.cast_type)
    elif args.method == "no_cast":
        cast_last(args.chunks, args.chunksize, base_type=args.base_type)
    else:
        raise ValueError("unexpected method")
