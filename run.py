#!/usr/bin/env python3
"""Provide example code to run ResNet-34 on Fashion MNIST dataset.
"""
import argparse
import logging
import logging.handlers
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

def parse_arguments():
    """Parse command line arguments.

    Returns:
        argparse.Namespace: The parsed arguments object.
    """
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument('--val_ratio', type=float, default=0.3,
                        help='The ratio of the validation set. (default: 0.3)')
    parser.add_argument('--batch_size', type=int, default=256,
                        help='The batch size to load the data. (default: 64)')
    parser.add_argument('--num_workers', type=int, default=4,
                        help=('The number of worker processes to use in '
                              'loading the dataset. (default: 4)'))
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

    return args

def prepare_logger():
    """Prepare formatted logger to stream and file.

    Returns:
        logging.Logger: The logger object.
    """
    # Prepare log directory.
    try:
        os.mkdir('logs')
    except FileExistsError:
        pass

    # Create logger and formatter.
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s %(message)s')

    # Create and attach stream handler.
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    # Create and attach file handler.
    file_handler = logging.handlers.TimedRotatingFileHandler(
        'logs/log.txt', when='d', encoding='utf-8')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger

def prepare_data(val_ratio, batch_size, num_workers):
    """Download Fashion MNIST dataset and prepare data loaders.

    Returns:
        tuple: A tuple of train, validation, and test data loaders.
    """
    # Define data preprocessing.
    transform = torchvision.transforms.Compose(
        [torchvision.transforms.ToTensor(),
         torchvision.transforms.Normalize((0.5, ), (0.5, ))])

    # Download and load the dataset.
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
    val_count = int(len(train_dataset) * val_ratio)
    train_indices = random_indices[val_count:]
    val_indices = random_indices[:val_count]

    # Create data loaders.
    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        sampler=torch.utils.data.sampler.SubsetRandomSampler(train_indices),
        num_workers=num_workers,
        drop_last=True)
    val_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        sampler=torch.utils.data.sampler.SubsetRandomSampler(val_indices),
        num_workers=num_workers,
        drop_last=False)
    test_loader = torch.utils.data.DataLoader(
        dataset=test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        drop_last=False)

    return train_loader, val_loader, test_loader

def prepare_model(learning_rate, momentum, checkpoint_file):
    """Prepare a ResNet-34 model with CrossEntropyLoss and SGD.

    Args:
        learning_rate (float): The learning rate for SGD.
        momentum (float): The momentum for SGD.
        checkpoint_file (str or None): If not `None`, the path of the
            checkpoint file to load.

    Returns:
        model.Model: The prepared model object.
    """
    # Load model.
    resnet = torchvision.models.resnet34()
    resnet.conv1 = torch.nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1,
                                   bias=False)
    resnet.avgpool = torch.nn.AvgPool2d(2)

    # Prepare loss function and optimizer.
    loss_function = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(resnet.parameters(),
                                lr=learning_rate,
                                momentum=momentum)

    # Wrap model object and load checkpoint file if provided.
    model = Model(resnet, loss_function, optimizer)
    if checkpoint_file:
        model.load(checkpoint_file)

    return model

def train(model, train_loader):
    """Train the model for an epoch with a pretty progress bar.

    Args:
        model (model.Model): The wrapped PyTorch model.
        train_loader (torch.utils.data.DataLoader): The train data loader.

    Returns:
        float: The mean training loss.
    """
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

    # Calculate mean training loss.
    mean_train_loss = torch.cat(loss_list).mean()

    return mean_train_loss

def validate(model, val_loader):
    """Validate the model with a pretty progress bar.

    Args:
        model (model.Model): The wrapped PyTorch model.
        train_loader (torch.utils.data.DataLoader): The train data loader.
        epoch (int): The current epoch number.

    Returns:
        float: The overall validation accuracy.
    """
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

    # Close progress bar.
    progress_bar.close()

    # Calculate validation accuracy.
    val_true = torch.cat(y_true_list)
    val_pred = torch.cat(y_pred_list)
    val_accuracy = (val_true == val_pred).sum() / len(val_true)

    return val_accuracy

def test(model, test_loader):
    """Validate the model with a pretty progress bar.

    Args:
        model (model.Model): The wrapped PyTorch model.
        train_loader (torch.utils.data.DataLoader): The train data loader.

    Returns:
        float: The overall test accuracy.
    """
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

    # Close progress bar.
    progress_bar.close()

    # Calculate test accuracy.
    test_true = torch.cat(y_true_list)
    test_pred = torch.cat(y_pred_list)
    test_accuracy = (test_true == test_pred).sum() / len(test_true)

    return test_accuracy

def main():
    """Train and validate ResNet-34 on Fashion MNIST dataset.
    """
    # Fix random seed.

    torch.manual_seed(0)

    # Prepare checkpoint directory.

    try:
        os.mkdir('checkpoints')
    except FileExistsError:
        pass

    # Make preparations.

    args = parse_arguments()
    logger = prepare_logger()
    train_loader, val_loader, test_loader = prepare_data(args.val_ratio,
                                                         args.batch_size,
                                                         args.num_workers)
    model = prepare_model(args.learning_rate, args.momentum, args.checkpoint_file)

    # Loop train-validate-save-log cycle per epoch.

    logger.info(' '.join(sys.argv))

    for epoch in range(args.num_epochs):

        # Train.
        mean_train_loss = train(model, train_loader)

        # Validate.
        val_accuracy = validate(model, val_loader)

        # Save.
        model.save('checkpoints/epoch{epoch}.pth'.format(epoch=epoch))

        # Log.
        message = ('[epoch {epoch}] mean training loss: {loss:.3f}, '
                   'validation accuracy: {accuracy:.3f}')
        message = message.format(epoch=epoch+1,
                                 loss=mean_train_loss,
                                 accuracy=val_accuracy)
        logger.info(message)

    # Test the resulting model, log results, and save.

    # Test.
    test_accuracy = test(model, test_loader)

    # Log.
    message = ('Final test accuracy: {accuracy:.3f}')
    message = message.format(accuracy=test_accuracy)
    logger.info(message)

    # Save.
    num = 1
    result_format = 'checkpoints/result{num}.pth'
    while os.path.isfile(result_format.format(num=num)):
        num += 1
    model.save(result_format.format(num=num))
    message = 'Final model saved as: checkpoints/result{num}.pth'
    logger.info(message.format(num=num))

if __name__ == '__main__':
    main()
