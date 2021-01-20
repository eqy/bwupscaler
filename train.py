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
PRINT_INTERVAL=100
MAX_STEPS=10000
GROUP=1

class Trainer():
    def __init__(self, train_path, val_path, model, group, loss_fn, lr, train_res=(256, 256), val_res=(512, 512), batch_size=16):
        self.model = model
        self.loss_fn = loss_fn
        self.lr = lr
        self.group = group
        self.batch_size = batch_size
        self.train_path = train_path
        self.val_path = val_path
        # TODO: revisit multiscale tunable parameter:
        self.scale = 4
        self.refiner = model(scale=self.scale, group=self.group)
        self.optimizer = optim.Adam(filter(lambda p: p.requires_grad, self.refiner.parameters()), self.lr)
        self.train_dataset = BWDataset(train_path, build_training_transform(train_res))
        self.train_loader = torch.utils.data.DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=NUM_WORKERS)

        self.val_dataset = BWDataset(val_path, build_validation_transform(val_res))
        # TODO: parametrizable source resolution
        samples_per_val = np.ceil(1920/val_res[0])*np.ceil(1080/val_res[1])
        val_batch_size = int(self.batch_size/samples_per_val)
        self.val_loader = torch.utils.data.DataLoader(self.val_dataset, batch_size=val_batch_size, shuffle=True, num_workers=NUM_WORKERS)
        self.step = 0
        self.print_interval = PRINT_INTERVAL
        self.max_steps = MAX_STEPS
        os.makedirs('checkpoints', exist_ok=True)

    def train(self):
        refiner = torch.nn.DataParallel(self.refiner)
        lr = self.lr
        refiner.train()
        while True:
            for hr, lr in self.train_loader:
                hr = hr.cuda()
                lr = lr.cuda()
                
                sr = refiner(lr, self.scale)
                loss = self.loss_fn(sr, hr)
                
                # TODO: switch to best practices for zero_grad
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm(refiner.parameters(), CLIP)
                self.optimizer.step() 
                self.step += 1
                if self.step % self.print_interval == 0:
                    print(loss)
                if self.step > self.max_steps:
                    break

def main():
    trainer = Trainer('data5/train', 'data5/val', model.carn_m.Net, GROUP, torch.nn.L1Loss(), 1e-4)
    trainer.train()

if __name__ == '__main__':
    main() 
