import argparse
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import torch
import torchvision
import torch.optim as optim
import os
import sys
import time
sys.path.insert(1, 'CARN-pytorch/carn')
import model.carn_m

from dataset import BWDataset, build_training_transform, build_validation_transform


NUM_WORKERS=4
CLIP=10.0
PRINT_INTERVAL=5000
MAX_STEPS=600000
DECAY=100000
GROUP=1

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

class Trainer():
    def __init__(self, train_path, val_path, model, group, loss_fn, learning_rate, decay=DECAY, train_res=(256, 256), val_res=(256, 256), batch_size=32, val_batch_size=32, model_name='carn', writer=None):
        self.model = model
        self.model_name = model_name
        self.loss_fn = loss_fn
        self.learning_rate = learning_rate
        self.group = group
        self.batch_size = batch_size
        self.val_batch_size = val_batch_size
        self.train_path = train_path
        self.val_path = val_path
        self.decay = decay
        self.writer = writer
        # TODO: revisit multiscale tunable parameter:
        self.scale = 4
        self.refiner = model(scale=self.scale, group=self.group)
        self.optimizer = optim.Adam(filter(lambda p: p.requires_grad, self.refiner.parameters()), self.learning_rate)
        self.train_res = train_res
        self.val_res = val_res
        self.train_dataset = BWDataset(train_path, build_training_transform(train_res))
        self.train_loader = torch.utils.data.DataLoader(self.train_dataset, batch_size=self.val_batch_size, shuffle=True, num_workers=NUM_WORKERS)

        self.val_dataset = BWDataset(val_path, build_validation_transform(val_res))
        # TODO: parametrizable source resolution
        samples_per_val = np.ceil(1920/val_res[0])*np.ceil(1080/val_res[1])
        actual_val_batch_size = max(int(self.val_batch_size/samples_per_val), 1)
        self.val_loader = torch.utils.data.DataLoader(self.val_dataset, batch_size=actual_val_batch_size, shuffle=True, num_workers=NUM_WORKERS)
        self.step = 0
        self.print_interval = PRINT_INTERVAL
        self.max_steps = MAX_STEPS
        os.makedirs('checkpoints', exist_ok=True)
        self.refiner = torch.nn.DataParallel(self.refiner)
        # statistics updated per batch
        self.loss = AverageMeter()
        self.train_data_time = AverageMeter()
        self.train_time = AverageMeter()
        self.val_data_time = AverageMeter()
        self.val_time = AverageMeter()
        self.psnr = AverageMeter()

    def decay_learning_rate(self):
        learning_rate = self.learning_rate * (0.5 ** (self.step//self.decay))
        return learning_rate

    def train(self):
        learning_rate = self.learning_rate
        while True:
            self.refiner.train()
            self.loss.reset()
            self.train_data_time.reset()
            self.train_time.reset()

            print(f"train... {len(self.train_loader)}")
            t = time.time()
            for i, (hr, lr) in enumerate(self.train_loader):
                hr = hr.cuda()
                lr = lr.cuda()
                self.train_data_time.update(time.time() - t, hr.size(0))

                sr = self.refiner(lr, self.scale)
                loss = self.loss_fn(sr, hr)
              
                # TODO: switch to best practices for zero_grad
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm(self.refiner.parameters(), CLIP)
                self.optimizer.step() 
                self.train_time.update(time.time() - t, hr.size(0))
                t = time.time()
                new_learning_rate = self.decay_learning_rate()
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = new_learning_rate
                
                self.loss.update(loss, hr.size(0))

                self.step += 1

                if self.step % self.print_interval == 0:
                    print(self.loss.avg, self.train_data_time.avg, self.train_time.avg)
                self.writer.add_scalar('Loss/train', loss, self.step)
                if self.step > self.max_steps:
                    break
            self.save('checkpoints', self.model_name)
            self.evaluate()

    def evaluate(self):
        print(f"validation... {len(self.val_loader)}")
        self.refiner.eval()
        self.psnr.reset()
        self.val_time.reset()
        self.val_data_time.reset()
        with torch.no_grad():
            t = time.time()
            for i, (hr, lr) in enumerate(self.val_loader):
                lr = lr.reshape((-1, 3, self.val_res[0]//self.scale, self.val_res[1]//self.scale))
                hr = hr.reshape((-1, 3, self.val_res[0], self.val_res[1]))
                lr = lr.cuda()
                hr = hr.cuda()
                self.val_data_time.update(time.time() - t, hr.size(0))
                sr = self.refiner(lr, self.scale)
                
                mpsnr = psnr(hr, sr)
                self.val_time.update(time.time() - t, hr.size(0))
                t = time.time()
                self.psnr.update(mpsnr, hr.size(0))
                if i % self.print_interval == 0:
                    print(self.psnr.avg, self.val_data_time.avg, self.val_time.avg)
                    self.writer.add_image('HRImage/val', torchvision.utils.make_grid(hr))
                    self.writer.add_image('LRImage/val', torchvision.utils.make_grid(lr))
                    self.writer.add_image('SRImage/val', torchvision.utils.make_grid(sr))
 
            print("done val", self.psnr.avg)
            self.writer.add_scalar('PSNR/val', self.psnr.avg, self.step)

    def save(self, checkpoint_dir, name):
        save_path = os.path.join(
            checkpoint_dir, "{}_{}.pth".format(name, self.step))
        torch.save(self.refiner.state_dict(), save_path)
        print("saved", save_path)

    def load(self, checkpoint_path):
        basename = os.path.basename(checkpoint_path)
        name, ext = os.path.splitext(basename)
        step = name.split('_')[-1]
        self.step = int(step)
        self.refiner.load_state_dict(torch.load(checkpoint_path))
        print("loaded model", checkpoint_path) 

def psnr(hr, sr):
    hr = hr.mul(255).clamp(0, 255) 
    sr = sr.mul(255).clamp(0, 255)
    # N C H W
    mmse = torch.mean((hr - sr)**2, dim=(0, 1, 2, 3))
    mpsnr = 255/(mmse + 1e-12)
    return mpsnr

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--name')
    parser.add_argument('--load-checkpoint')
    parser.add_argument('--decay', default=50000, type=int)
    args = parser.parse_args()
    writer = SummaryWriter(comment=f'{args.name}_decay{args.decay}')
    trainer = Trainer('data5/train', 'data5/val', model.carn_m.Net, GROUP, torch.nn.L1Loss(), 1e-5, batch_size=48, writer=writer, model_name=args.name, decay=args.decay)
    if (args.load_checkpoint is not None):
        trainer.load(args.load_checkpoint)
    trainer.train()

if __name__ == '__main__':
    main() 
