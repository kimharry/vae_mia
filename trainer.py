from torch.autograd import Variable
from torch import optim
import torch

import shutil
from tqdm import tqdm
import numpy as np
from random import choice

from resnet32 import resnet32_for_mia as resnet32


class Trainer:
    def __init__(self, targets, model, loss, train_loader, test_loader, args):
        self.targets = targets

        self.model = model
        self.args = args
        self.args.start_epoch = 0

        self.train_loader = train_loader
        self.test_loader = test_loader

        # Loss function and Optimizer
        self.loss = loss
        self.optimizer = self.get_optimizer()

        # Model Loading
        if args.resume:
            if len(args.targets) == 1:
                self.load_checkpoint(self.args.resume_from + "_total.pth.tar")
            else:
                self.load_checkpoint(self.args.resume_from + "_split.pth.tar")

    def train(self):
        self.model.train()
        best_loss = 10000000
        for epoch in range(self.args.start_epoch, self.args.num_epochs):
            loss_list = []
            print("epoch {}...".format(epoch))
            for batch_idx, (data, _) in enumerate(tqdm(self.train_loader)):
                if self.args.cuda:
                    data = data.cuda()
                data = Variable(data)
                self.optimizer.zero_grad()
                temp_target = choice(self.targets)
                target_output = temp_target(data).view(-1, 64, 8, 8)
                recon_batch, mu, logvar = self.model(target_output)
                loss = self.loss(recon_batch, data, mu, logvar)
                loss.backward()
                self.optimizer.step()
                loss_list.append(loss.item())

            epoch_loss = np.mean(loss_list)
            print("epoch {}: - loss: {}".format(epoch, epoch_loss))
            new_lr = self.adjust_learning_rate(epoch)
            print('learning rate:', new_lr)

            if epoch_loss < best_loss:
                if len(args.targets) == 1:
                    state = {
                        'epoch': epoch + 1,
                        'state_dict': self.model.state_dict(),
                        'optimizer': self.optimizer.state_dict(),
                    }
                    torch.save(state, self.args.resume_from + "_total.pth.tar")
                else:
                    state = {
                        'epoch': epoch + 1,
                        'state_dict': self.model.state_dict(),
                        'optimizer': self.optimizer.state_dict(),
                    }
                    torch.save(state, self.args.resume_from + "_split.pth.tar")
                best_loss = epoch_loss

            if epoch % self.args.test_every == 0:
                self.test(epoch)

    def test(self, cur_epoch):
        print('testing...')
        self.model.eval()
        test_loss = 0
        for i, (data, _) in enumerate(self.test_loader):
            if self.args.cuda:
                data = data.cuda()
            data = Variable(data, volatile=True)
            temp_target = choice(self.targets)
            target_output = temp_target(data).view(-1, 64, 8, 8)
            recon_batch, mu, logvar = self.model(target_output)
            test_loss += self.loss(recon_batch, data, mu, logvar).item()

        test_loss /= len(self.test_loader.dataset)
        print('====> Test set loss: {:.4f}'.format(test_loss))
        self.model.train()

    def test_on_trainings_set(self):
        print('testing...')
        self.model.eval()
        test_loss = 0
        for i, (data, _) in enumerate(self.train_loader):
            if self.args.cuda:
                data = data.cuda()
            data = Variable(data, volatile=True)
            temp_target = choice(self.targets)
            target_output = temp_target(data).view(-1, 64, 8, 8)
            recon_batch, mu, logvar = self.model(target_output)
            test_loss += self.loss(recon_batch, data, mu, logvar).item()

        test_loss /= len(self.test_loader.dataset)
        print('====> Test on training set loss: {:.4f}'.format(test_loss))
        self.model.train()

    def get_optimizer(self):
        return optim.Adam(self.model.parameters(), lr=self.args.learning_rate,
                          weight_decay=self.args.weight_decay)

    def adjust_learning_rate(self, epoch):
        """Sets the learning rate to the initial LR multiplied by 0.98 every epoch"""
        # learning_rate = self.args.learning_rate * (self.args.learning_rate_decay ** epoch)
        if epoch in self.args.learning_rate_decay_epochs:
            learning_rate = self.args.learning_rate * (self.args.gamma ** (self.args.learning_rate_decay_epochs.index(epoch) + 1))
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = learning_rate
        return learning_rate

    def load_checkpoint(self, filename):
        filename = self.args.checkpoint_dir + filename
        try:
            print("Loading checkpoint '{}'".format(filename))
            checkpoint = torch.load(filename)
            self.args.start_epoch = checkpoint['epoch']
            self.model.load_state_dict(checkpoint['state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            print("Checkpoint loaded successfully from '{}' at (epoch {})\n"
                  .format(self.args.checkpoint_dir, checkpoint['epoch']))
        except:
            print("No checkpoint exists from '{}'. Skipping...\n".format(self.args.checkpoint_dir))
