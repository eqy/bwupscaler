import clip
from models import rcan
import rcan_options
import train
import torch
import torchvision


def build_upscale_fn(trainer):
    def upscale_fn(lr):
        lr = lr.cuda()
        sr = trainer.run_model(lr)
        return sr
    return upscale_fn


def main():
    source = clip.Clip("testdata/Boxer's Perfect SCV Rush-Jen46qkZVNI.mp4")
    trainer = train.Inferencer(rcan.make_model(rcan_options.rcan_options()))
    trainer.load('checkpoints/rcan_test_noshift_257424.pth')
    trainer.inference_mode()
    upscale_fn = build_upscale_fn(trainer)
    source.upscale("test_nopreproc.mp4", upscale_fn)


if __name__ == '__main__':
    main()
