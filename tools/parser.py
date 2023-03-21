import argparse
import hubconf


def get_default_parser():
    model_names = [
        name
        for name in hubconf.__dict__
        if "in1k" in name and callable(hubconf.__dict__[name])
    ]

    parser = argparse.ArgumentParser(description="PyTorch ImageNet Training")
    parser.add_argument("data", metavar="DIR", help="path to dataset")
    parser.add_argument(
        "-m",
        "--model",
        metavar="MODEL",
        default="vit_b16_in1k",
        choices=model_names,
        help="model name: " + " | ".join(model_names) + " (default: hourglass_vit_b16_in1k)",
    )
    parser.add_argument(
        "-r", "--resolution", default=224, type=int, help="input resolution of the images"
    )
    parser.add_argument(
        "-j",
        "--workers",
        default=4,
        type=int,
        metavar="N",
        help="number of data loading workers (default: 4)",
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
    parser.add_argument(
        "-p",
        "--print-freq",
        default=10,
        type=int,
        metavar="N",
        help="print frequency (default: 10)",
    )
    parser.add_argument(
        "--world-size",
        default=-1,
        type=int,
        help="number of nodes for distributed training",
    )
    parser.add_argument(
        "--rank", default=-1, type=int, help="node rank for distributed training"
    )
    parser.add_argument(
        "--dist-url",
        default="tcp://224.66.41.62:23456",
        type=str,
        help="url used to set up distributed training",
    )
    parser.add_argument(
        "--dist-backend", default="nccl", type=str, help="distributed backend"
    )
    parser.add_argument("--gpu", default=None, type=int, help="GPU id to use.")
    parser.add_argument(
        "--multiprocessing-distributed",
        action="store_true",
        help="Use multi-processing distributed training to launch "
             "N processes per node, which has N GPUs. This is the "
             "fastest way to use PyTorch for either single node or "
             "multi node data parallel training",
    )

    add_model_args(parser)

    return parser

def add_model_args(parser):
    add_expedit_vit_args(parser)
    add_tome_args(parser)
    add_evit_args(parser)
    add_act_vit_args(parser)
    add_smyrf_vit_args(parser)
    add_ats_vit_args(parser)
    add_evo_vit_args(parser)

def add_expedit_vit_args(parser):
    parser.add_argument(
        "-l",
        "--clustering_location",
        type=int,
        default=-1,
        help="location of clustering, ranging from [0, num of layers of transformer)"
    )
    parser.add_argument(
        "-n",
        "--num_cluster",
        type=int,
        default=1000,
        help="num of clusters, no more than total number of features"
    )
    parser.add_argument(
        "--cluster_iters",
        type=int,
        default=5,
        help="num of iterations in clustering"
    )
    parser.add_argument(
        "--cluster_temperture",
        type=float,
        default=1.,
        help="temperture in clustering"
    )
    parser.add_argument(
        "--cluster_window_size",
        type=int,
        default=5,
        help="window size in clustering"
    )


def add_tome_args(parser):
    parser.add_argument(
        "--tome_r",
        type=int,
        default=0,
        help="num of tokens of each step of merging, ranging from [0, num of tokens / 2)"
    )
    
    
def add_evit_args(parser):
    parser.add_argument("--base_keep_rate", type=float, default=0.7)
    parser.add_argument("--drop_loc", type=int, nargs='+', default=[3, 6, 9])
    parser.add_argument("--fuse_token", action='store_true', default=False)


def add_act_vit_args(parser):
    parser.add_argument(
        "--act_plug_in_index",
        type=int,
        default=0,
        help='act attention plug in index'
    )
    parser.add_argument(
        "--act_plug_out_index",
        type=int,
        default=-1,
        help='act attention plug out index'
    )
    parser.add_argument(
        "--act_group_q",
        type=bool,
        default=True,
        help='act query use grouping'
    )
    parser.add_argument(
        "--act_group_k",
        type=bool,
        default=False,
        help='act query use grouping'
    )
    parser.add_argument(
        "--act_q_hashes",
        type=int,
        default=32,
        help='act query hash times'
    )
    parser.add_argument(
        "--act_k_hashes",
        type=int,
        default=32,
        help='act key hash times'
    )

def add_smyrf_vit_args(parser):
    parser.add_argument(
        "--smyrf_plug_in_index",
        type=int,
        default=0,
        help='smyrf attention plug in index'
    )
    parser.add_argument(
        "--smyrf_plug_out_index",
        type=int,
        default=-1,
        help='smyrf attention plug out index'
    )
    parser.add_argument(
        "--smyrf_n_hashes",
        type=int,
        default=32,
        help='smyrf hash times'
    )
    parser.add_argument(
        "--smyrf_q_cluster_size",
        type=int,
        default=64,
        help='smyrf query cluster size'
    )
    parser.add_argument(
        "--smyrf_k_cluster_size",
        type=int,
        default=64,
        help='smyrf key cluster size'
    )

def add_ats_vit_args(parser):
    parser.add_argument(
        "--ats_blocks_indexes",
        type=int,
        nargs='*',
        default=[],
        help='indexes of using ats blocks'
    )
    parser.add_argument(
        "--ats_num_tokens",
        type=int,
        default=197,
        help='num of sampled tokens in ats blocks'
    )
    parser.add_argument(
        "--ats_drop_tokens",
        action="store_true",
        default=False,
        help='whether to drop tokens in ats blocks'
    )

def add_evo_vit_args(parser):
    parser.add_argument(
        "--evo_prune_ratio",
        type=float,
        # nargs='*',
        default=0.5,
        help='ratio of pruning tokens in evo blocks'
    )
    parser.add_argument(
        "--evo_prune_location",
        type=int,
        # nargs='*',
        default=4,
        help='starting location of pruning in evo blocks'
    )

def get_model_args(args):
    if 'hourglass_vit' in args.model:
        return {
            "clustering_location":  args.clustering_location,
            "num_cluster":          args.num_cluster,
            "cluster_iters":        args.cluster_iters,
            "cluster_temperture":   args.cluster_temperture,
            "cluster_window_size":  args.cluster_window_size,
        }
    elif 'tome' in args.model:
        return {
            "tome_r":               args.tome_r,
        }
    elif 'e_vit' in args.model:
        return {
            "base_keep_rate":       args.base_keep_rate,
            "drop_loc":             args.drop_loc,
            "fuse_token":           args.fuse_token,
        }
    elif 'act' in args.model:
        return {
            "act_plug_in_index": args.act_plug_in_index,
            "act_plug_out_index": args.act_plug_out_index,
            "act_group_q": args.act_group_q,
            "act_group_k": args.act_group_k,
            "act_q_hashes": args.act_q_hashes,
            "act_k_hashes": args.act_k_hashes,
        }
    elif 'smyrf' in args.model:
        return {
            "smyrf_plug_in_index": args.smyrf_plug_in_index,
            "smyrf_plug_out_index": args.smyrf_plug_out_index,
            "smyrf_n_hashes": args.smyrf_n_hashes,
            "smyrf_q_cluster_size": args.smyrf_q_cluster_size,
            "smyrf_k_cluster_size": args.smyrf_k_cluster_size,
        }
    elif 'ats' in args.model:
        return {
            "ats_blocks":           args.ats_blocks_indexes,
            "num_tokens":           args.ats_num_tokens,
            "drop_tokens":          args.ats_drop_tokens   
        }
    elif 'evo' in args.model:
        return {
            "prune_ratio":          args.evo_prune_ratio,
            "prune_location":       args.evo_prune_location,
        }
    return dict()
