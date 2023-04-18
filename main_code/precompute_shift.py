import time
import pandas as pd
import argparse
from sub_resnet152 import calibrate_shift, list2str


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--nb_blocks", type=str, default="1,1,1,1",
                        help="""comma-separated list of 4 integers. 
                        each element cannot surpass 3,8,36,3""")
    args = parser.parse_args()
    
    args.nb_blocks = [int(xx) for xx in args.nb_blocks.split(",")]
    assert len(args.nb_blocks) == 4
    assert args.nb_blocks[0] >= 1 and args.nb_blocks[0] <= 3
    assert args.nb_blocks[1] >= 1 and args.nb_blocks[1] <= 8
    assert args.nb_blocks[2] >= 1 and args.nb_blocks[2] <= 36
    assert args.nb_blocks[3] >= 1 and args.nb_blocks[3] <= 3

    return args


if __name__ == "__main__":
    
    t0 = time.time()

    args = parse_arguments()
    
    nb_blocks = args.nb_blocks
    
    nb_classes_list = [62, 20, 19]
    
    divergence_idx_list = [15, 51]
    
    nb_branch_list = [1, 2, 4, 8, 16, 32, 64, 128, 256]

    df = []
    for nb_classes in nb_classes_list:
        for divergence_idx in divergence_idx_list:
            for nb_branch in nb_branch_list:
                
                print("Computing shift for nb_classes={}, divergence_idx={}, nb_branch={}".format(
                    nb_classes, divergence_idx, nb_branch))
                try:
                    shift, diverged_planes_per_block, percent = calibrate_shift(
                                            divergence_idx=divergence_idx,
                                            nb_branch=nb_branch, nb_classes=nb_classes, 
                                            verbose=False, 
                                            return_details=True,
                                            nb_blocks=nb_blocks)
                    
                    print("shift={}, percent={}, diverged_planes_per_block={}".format(
                        shift, percent, diverged_planes_per_block
                    ))
                except Exception as e:
                    print("Failed! nb_classes={}, divergence_idx={}, nb_branch={}: {}".format(
                        nb_classes, divergence_idx, nb_branch, e))
                    # - 99999 is a marker for failed configuration, see sub_resnet152.py
                    shift = - 99999
                    diverged_planes_per_block = [- 1, - 1, - 1, - 1]
                    percent = - 1
                print()
                df.append([nb_classes, divergence_idx, nb_branch, shift, 
                           percent,
                           diverged_planes_per_block[0],
                           diverged_planes_per_block[1],
                           diverged_planes_per_block[2],
                           diverged_planes_per_block[3]])
    
    df = pd.DataFrame(
        df, columns=["nb_classes", "divergence_idx", "nb_branch", "shift", 
                     "percent",
                     "diverged_planes_per_block_0",
                     "diverged_planes_per_block_1",
                     "diverged_planes_per_block_2",
                     "diverged_planes_per_block_3"])
    
    df.to_csv("precomputed_shift_lookup_table_{}.csv".format(list2str(nb_blocks)))
    
    
    
    print(df)
    
    print("Done in {:.2f} s.".format(time.time() - t0))




