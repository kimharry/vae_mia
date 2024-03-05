from __future__ import print_function

from torch.autograd import Variable

import torch.backends.cudnn as cudnn
import torch.nn.init as init
import torch.utils.data

from data_loaders.cifar10_data_loader import CIFAR10DataLoader
from loss import Loss
from vae import VAE
from trainer import Trainer
from utils.utils import *
from utils.weight_initializer import Initializer

import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

from resnet32 import resnet32

def plot_images(original_images, reconstructed_images, title, filename):
    num_images = original_images.shape[0]
    num_comparisons = len(reconstructed_images)
    fig, axs = plt.subplots(num_comparisons+1, num_images, figsize=(num_images*num_comparisons+1, 4))

    image_shape = (3, 32, 32)

    for i in range(num_images):
        axs[0, i].imshow(np.clip(np.transpose(original_images[i].reshape(image_shape), (1, 2, 0)), 0, 1))
        axs[0, i].axis('off')
        for j in range(1, num_comparisons+1):
            axs[j, i].imshow(np.clip(np.transpose(reconstructed_images[j-1][i].reshape(image_shape), (1, 2, 0)), 0, 1))
            axs[j, i].axis('off')

    fig.suptitle(title)

    now = datetime.now() 
    current_time = now.strftime("%Y-%m-%d_%H-%M-%S")
    filename_with_time = f"{filename}_{current_time}.png"

    plt.savefig(filename_with_time)
    plt.show()


def main():
    # Parse the JSON arguments
    args = parse_args()

    # Create the experiment directories
    args.summary_dir, args.checkpoint_dir = create_experiment_dirs(
        args.experiment_dir)

    model = VAE()

    # to apply xavier_uniform:
    Initializer.initialize(model=model, initialization=init.xavier_uniform, gain=init.calculate_gain('relu'))

    loss = Loss()

    if torch.cuda.is_available():
        model.cuda()
        loss.cuda()
        cudnn.enabled = True
        cudnn.benchmark = True

    print("Loading Data...")
    data = CIFAR10DataLoader(args)
    print("Data loaded successfully\n")

    target = resnet32()
    pretrained_state_dict = torch.load('resnet32_cifar10.pth')
    target.load_state_dict(pretrained_state_dict['net'])
    target.to("cuda")

    trainer = Trainer(target, model, loss, data.train_loader, data.test_loader, args)

    if args.to_train:
        try:
            print("Training...")
            trainer.train()
            print("Training Finished\n")
        except KeyboardInterrupt:
            print("Training had been Interrupted\n")

    if args.to_test:
        print("Testing on training data...")
        trainer.test_on_trainings_set()
        print("Testing Finished\n")

    if args.generate_result:
        temp_data, _ = next(iter(data.test_loader))
        if args.cuda:
            temp_data = temp_data.cuda()
        temp_data = Variable(temp_data)
        target_output = target(temp_data).view(-1, 10, 1, 1)
        [outputs, _, _] = model(target_output)
        outputs = outputs.view(-1, 3, 32, 32)
        outputs = outputs.detach().cpu().numpy()

        plot_images(temp_data.detach().cpu().numpy(), [outputs], 'VAE', 'VAE')


if __name__ == "__main__":
    main()
