#!/usr/bin/env python3
"""Provide example code to run ResNet-34 on Fashion MNIST dataset.
"""
import argparse
import logging
import os
import sys

import torch
import torch.utils.data
import torchvision
import torchvision.datasets
import torchvision.models
import torchvision.transforms
import tqdm

from model import Model

def main():
    """Train and validate ResNet-34 on Fashion MNIST dataset.
    """
    # Enable logging to STDOUT.
    logging.basicConfig(format='%(asctime)s %(message)s',
                        level=logging.INFO)

    # Parse arguments.
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--batch_size', type=int, default=256,
                        help='The batch size to load the data. (default: 64)')
    parser.add_argument('--num_workers', type=int, default=4,
                        help=('The number of worker processes to use in '
                              'loading the dataset. (default: 4)'))
    parser.add_argument('--val_ratio', type=float, default=0.3,
                        help='The ratio of the validation set. (default: 0.3)')
    parser.add_argument('--num_epochs', type=int, default=30,
                        help=('The number of training epochs to run. (default:'
                              '30)'))
    parser.add_argument('--learning_rate', type=float, default=0.1,
                        help='The learning rate for SGD. (default: 0.1)')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='The momentum for SGD. (default: 0.9)')
    parser.add_argument('--checkpoint_file',
                        help='The path of the checkpoint file to load')
    args = parser.parse_args(sys.argv[1:])

    # Fix random seed.  torch.manual_seed(0)

    # Create checkpoint directory.
    try:
        os.mkdir('checkpoints')
    except FileExistsError:
        pass

    logging.info('Preparing data..')

    # Download and load the dataset.
    transform = torchvision.transforms.Compose(
        [torchvision.transforms.ToTensor(),
         torchvision.transforms.Normalize((0.5, ), (0.5, ))])
    train_dataset = torchvision.datasets.FashionMNIST(root='./data',
                                                      train=True,
                                                      download=True,
                                                      transform=transform)
    test_dataset = torchvision.datasets.FashionMNIST(root='./data',
                                                     train=True,
                                                     download=True,
                                                     transform=transform)

    # Create shuffled indices, and split into given ratio.
    random_indices = torch.randperm(len(train_dataset))
    val_count = int(len(train_dataset) * args.val_ratio)
    train_indices = random_indices[val_count:]
    val_indices = random_indices[:val_count]

    # Create data loaders.
    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=args.batch_size,
        sampler=torch.utils.data.sampler.SubsetRandomSampler(train_indices),
        num_workers=args.num_workers,
        drop_last=True)
    val_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=args.batch_size,
        sampler=torch.utils.data.sampler.SubsetRandomSampler(val_indices),
        num_workers=args.num_workers,
        drop_last=False)
    test_loader = torch.utils.data.DataLoader(
        dataset=test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        drop_last=False)

    logging.info('Loading model..')

    # Prepare model.
    resnet = torchvision.models.resnet34()
    resnet.conv1 = torch.nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1,
                                   bias=False)
    resnet.avgpool = torch.nn.AvgPool2d(2)

    # Prepare loss function and optimizer.
    loss_function = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(resnet.parameters(),
                                lr=args.learning_rate,
                                momentum=args.momentum)

    # Wrap model object and load checkpoint file if provided.
    model = Model(resnet, loss_function, optimizer)
    if args.checkpoint_file:
        model.load(checkpoint_file)

    logging.info('Training start!')

    for epoch in range(args.num_epochs):

        # Create progress bar.
        progress_bar = tqdm.tqdm(total=len(train_loader),
                                 unit='batch',
                                 desc='[train] batch loss: 0.000',
                                 leave=False)

        # Create an empty list to gather batch losses.
        loss_list = []

        # Loop through training batches.
        for i, (x, y) in enumerate(train_loader):

            # Forward and backward.
            batch_loss = model.train_batch(x, y)

            # Gather batch losses.
            loss_list.append(batch_loss)

            # Update progress bar.
            progress_bar.update(1)
            progress_bar.set_description(
                '[train] batch loss: {loss:.3f}'.format(loss=batch_loss[0]))

        # Close progress bar.
        progress_bar.close()

        # Create progress bar.
        progress_bar = tqdm.tqdm(total=len(val_loader),
                                 unit='batch',
                                 desc='[validate] batch accuracy: 0.000',
                                 leave=False)

        # Create empty lists to gather true and inferred labels.
        y_true_list = []
        y_pred_list = []

        # Loop through validation batches.
        for i, (x, y) in enumerate(val_loader):

            # Forward.
            y_pred = model.infer_batch(x)

            # Gather true and inferred labels.
            y_true_list.append(y)
            y_pred_list.append(y_pred)

            # Update progress bar.
            accuracy = (y == y_pred).sum() / len(y)
            progress_bar.update(1)
            progress_bar.set_description(
                '[validate] batch accuracy: {accuracy:.3f}'.format(
                    accuracy=accuracy))

        # Calculate statistics.
        mean_training_loss = torch.cat(loss_list).mean()
        val_true = torch.cat(y_true_list)
        val_pred = torch.cat(y_pred_list)
        val_accuracy = (val_true == val_pred).sum() / len(val_true)

        # Close progress bar and log statistics.
        progress_bar.close()
        message = ('[epoch {epoch}] mean training loss: {loss:.3f}, '
                   'validation accuracy: {accuracy:.3f}')
        message = message.format(epoch=epoch+1,
                                 loss=mean_training_loss,
                                 accuracy=val_accuracy)
        logging.info(message)

        # Save checkpoint.
        model.save('checkpoints/epoch{epoch}.pth'.format(epoch=epoch))

    logging.info('Testing start!')

    # Create empty lists to gather true and inferred labels.
    y_true_list = []
    y_pred_list = []

    # Create progress bar.
    progress_bar = tqdm.tqdm(total=len(test_loader),
                             unit='batch',
                             desc='[test] batch accuracy: 0.000',
                             leave=False)

    # Loop through test batches.
    for i, (x, y) in enumerate(test_loader):

        # Forward.
        y_pred = model.infer_batch(x)

        # Gather true and inferred labels.
        y_true_list.append(y)
        y_pred_list.append(y_pred)

        # Update progress bar.
        accuracy = (y == y_pred).sum() / len(y)
        progress_bar.update(1)
        progress_bar.set_description(
            '[test] batch accuracy: {accuracy:.3f}'.format(accuracy=accuracy))

    # Calculate statistics.
    test_true = torch.cat(y_true_list)
    test_pred = torch.cat(y_pred_list)
    test_accuracy = (test_true == test_pred).sum() / len(test_true)

    # Close progress bar and log statistics.
    progress_bar.close()
    message = ('Final test accuracy: {accuracy:.3f}')
    message = message.format(accuracy=test_accuracy)
    logging.info(message)

if __name__ == '__main__':
    main()
