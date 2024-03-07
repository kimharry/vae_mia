import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader

from model import resnet
import argparse
from tensorboardX import SummaryWriter


parser = argparse.ArgumentParser(description='cifar10 classification models')
parser.add_argument('--lr', default=0.1, help='')
parser.add_argument('--resume', default=None, help='')
parser.add_argument('--batch_size', default=128, help='')
parser.add_argument('--batch_size_test', default=100, help='')
parser.add_argument('--num_worker', default=4, help='')
parser.add_argument('--logdir', type=str, default='logs', help='')
args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'

print('==> Preparing data..')
transforms_train = transforms.Compose([
	transforms.RandomCrop(32, padding=4),
	transforms.RandomHorizontalFlip(),
	transforms.ToTensor(),
	transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])
transforms_test = transforms.Compose([
	transforms.ToTensor(),
	transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

dataset_train = CIFAR10(root='../data', train=True, 
						download=True, transform=transforms_train)
dataset_test = CIFAR10(root='../data', train=False, 
					   download=True, transform=transforms_test)
train_loader = DataLoader(dataset_train, batch_size=args.batch_size, 
	                      shuffle=True, num_workers=args.num_worker)
test_loader = DataLoader(dataset_test, batch_size=args.batch_size_test, 
	                     shuffle=False, num_workers=args.num_worker)

# there are 10 classes so the dataset name is cifar-10
classes = ('plane', 'car', 'bird', 'cat', 'deer', 
	       'dog', 'frog', 'horse', 'ship', 'truck')

print('==> Making model..')

net = resnet()
net = net.to(device)
num_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
print('The number of parameters of model is', num_params)
# print(net)

if args.resume is not None:
	checkpoint = torch.load('./save_model/' + args.resume)
	net.load_state_dict(checkpoint['net'])

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.1,
	                  momentum=0.9, weight_decay=1e-4)

decay_epoch = [32000, 48000]
step_lr_scheduler = lr_scheduler.MultiStepLR(optimizer, 
								 milestones=decay_epoch, gamma=0.1)
writer = SummaryWriter(args.logdir)


def train(epoch, global_steps):
	net.train()

	train_loss = 0
	correct = 0
	total = 0

	for batch_idx, (inputs, targets) in enumerate(train_loader):
		global_steps += 1
		step_lr_scheduler.step()
		inputs = inputs.to(device)
		targets = targets.to(device)
		outputs = net(inputs)
		loss = criterion(outputs, targets)

		optimizer.zero_grad()
		loss.backward()
		optimizer.step()

		train_loss += loss.item()
		_, predicted = outputs.max(1)
		total += targets.size(0)
		correct += predicted.eq(targets).sum().item()
		
	acc = 100 * correct / total
	print('train epoch : {} [{}/{}]| loss: {:.3f} | acc: {:.3f}'.format(
		   epoch, batch_idx, len(train_loader), train_loss/(batch_idx+1), acc))

	writer.add_scalar('log/train error', 100 - acc, global_steps)
	return global_steps


# def test(epoch, best_acc, global_steps):
def test(epoch, global_steps):
	net.eval()

	test_loss = 0
	correct = 0
	total = 0

	with torch.no_grad():
		for batch_idx, (inputs, targets) in enumerate(test_loader):
			inputs = inputs.to(device)
			targets = targets.to(device)
			outputs = net(inputs)
			loss = criterion(outputs, targets)

			test_loss += loss.item()
			_, predicted = outputs.max(1)
			total += targets.size(0)
			correct += predicted.eq(targets).sum().item()
	
	acc = 100 * correct / total
	print('test epoch : {} [{}/{}]| loss: {:.3f} | acc: {:.3f}'.format(
		   epoch, batch_idx, len(test_loader), test_loss/(batch_idx+1), acc))

	writer.add_scalar('log/test error', 100 - acc, global_steps)
    
	# if acc > best_acc:
		# print('==> Saving model..')
		# state = {
		#     'net': net.state_dict(),
		#     'acc': acc,
		#     'epoch': epoch,
		# }
		# if not os.path.isdir('save_model'):
		#     os.mkdir('save_model')
		# torch.save(state, './save_model/ckpt.pth')
		# best_acc = acc

	return acc, test_loss/(batch_idx+1)


if __name__=='__main__':
	best_acc = 0
	acc = 0
	epoch = 0
	global_steps = 0
	# best_loss = 100
	# loss = 0

	# for early stopping
	# early_stopping = False
	# patience = 30
	# latest_accs = [0 for i in range(patience)]
	# latest_losses = [0 for i in range(patience)]
	
	if args.resume is not None:
		test(epoch=0, best_acc=0)
	else:
		while True:
			epoch += 1
			global_steps = train(epoch, global_steps)
			# acc = test(epoch, global_steps)
			acc, loss = test(epoch, global_steps)
			if acc > best_acc:
				best_acc = acc
			# 	# print('==> Saving model..')
			# 	state = {
			# 	    'net': net.state_dict(),
			# 	    'acc': best_acc,
			# 	    'epoch': epoch,
			# 	}
			# 	if not os.path.isdir('save_model'):
			# 	    os.mkdir('save_model')
			# 	torch.save(state, './save_model/ckpt_240113_4.pth')
			# if loss < best_loss:
			# 	best_loss = loss

			# del latest_accs[0]
			# latest_accs.append(acc)
			# if best_acc == latest_accs[0]:
			# del latest_losses[0]
			# latest_losses.append(loss)
			# if best_loss == latest_losses[0]:
			# 	early_stopping = True
			
			# if global_steps >= 64000:
			if epoch >= 85:
				state = {
				    'net': net.state_dict(),
				    'acc': best_acc,
				    'epoch': epoch,
				}
				if not os.path.isdir('save_model'):
				    os.mkdir('save_model')
				torch.save(state, './save_model/ckpt_240114_7.pth')
				print("training finished!")
				break