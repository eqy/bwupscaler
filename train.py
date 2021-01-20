import numpy as np
import torch
import torch.optim as optim
import os
import sys
sys.path.insert(1, 'CARN-pytorch/carn')
import model.carn_m

from dataset import BWDataset, build_training_transform, build_validation_transform


NUM_WORKERS=4
CLIP=10.0
PRINT_INTERVAL=1000
MAX_STEPS=10000
DECAY=150000
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
    def __init__(self, train_path, val_path, model, group, loss_fn, learning_rate, decay=DECAY, train_res=(256, 256), val_res=(256, 256), batch_size=16, val_batch_size=8):
        self.model = model
        self.loss_fn = loss_fn
        self.learning_rate = learning_rate
        self.group = group
        self.batch_size = batch_size
        self.val_batch_size = val_batch_size
        self.train_path = train_path
        self.val_path = val_path
        self.decay = decay
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
        learning_rate = self.learning_rate * (0.5 ** (self.step*self.batch_size//self.decay))
        return learning_rate

    def train(self):
        learning_rate = self.learning_rate
        while True:
            self.refiner.train()
            for i, (hr, lr) in enumerate(self.train_loader):
                hr = hr.cuda()
                lr = lr.cuda()
                sr = self.refiner(lr, self.scale)
                loss = self.loss_fn(sr, hr)
              
                # TODO: switch to best practices for zero_grad
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm(self.refiner.parameters(), CLIP)
                self.optimizer.step() 

                new_learning_rate = self.decay_learning_rate()
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = new_learning_rate
 
                self.step += 1
                if self.step % self.print_interval == 0:
                    print(loss)
                if self.step > self.max_steps:
                    break
            self.evaluate()

    def evaluate(self):
        self.refiner.eval()
        with torch.no_grad():
            for i, (hr, lr) in enumerate(self.val_loader):
                lr = lr.reshape((-1, 3, self.val_res[0]//self.scale, self.val_res[1]//self.scale))
                hr = hr.reshape((-1, 3, self.val_res[0], self.val_res[1]))
                lr = lr.cuda()
                hr = hr.cuda()
                sr = self.refiner(lr, self.scale)
                
                mpsnr = psnr(hr, sr)
                self.psnr.update(mpsnr, hr.size(0))
            print(self.psnr.avg)

def psnr(hr, sr):
    hr = hr.mul(255).clamp(0, 255) 
    sr = sr.mul(255).clamp(0, 255)
    # N C H W
    mmse = torch.mean((hr - sr)**2, dim=(0, 1, 2, 3))
    print(mmse)
    mpsnr = 255/(mmse + 1e-12)
    return mpsnr

def main():
    trainer = Trainer('data5/train', 'data5/val', model.carn_m.Net, GROUP, torch.nn.L1Loss(), 1e-5, batch_size=32)
    trainer.train()

if __name__ == '__main__':
    main() 
