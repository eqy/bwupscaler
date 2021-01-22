from argparse import Namespace

def rcan_options():
    dummy = Namespace(scale=[4], n_resgroups=10, n_resblocks=20, n_feats=64, patch_size=256, reduction=16, rgb_range=255, n_colors=3, res_scale=1)
    return dummy 
