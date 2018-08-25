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
from torchvision.datasets import FashionMNIST
import torchvision.models
import torchvision.transforms
import tqdm


def get_args():
    """Parse and return command line arguments.

    Returns:
        argparse.Namespace: The parsed arguments object.
    """
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument('--batch_size', type=int, default=64,
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
    parser.add_argument('--checkpoint',
                        help='The path of the checkpoint file to load')
    parser.add_argument('--cuda', default=False, action='store_true',
                        help='Use GPU if available.')

    args = parser.parse_args(sys.argv[1:])

    return args


def get_logger():
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


def get_data(batch_size, num_workers):
    """Download Fashion MNIST dataset and wrap with loaders.

    Returns:
        tuple: A tuple of train, validation, and test data loaders.
    """
    # Define data preprocessing.
    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.5, ), (0.5, )),
    ])

    # Download and load the FashinMNIST data.
    data = FashionMNIST(root='./data',
                        train=True,
                        download=True,
                        transform=transform)
    data_test = FashionMNIST(root='./data',
                             train=False,
                             download=True,
                             transform=transform)

    # Split training and validation data.
    len_train = int(len(data) * 0.8)
    len_val = len(data) - len_train
    data_train, data_val = torch.utils.data.random_split(
        data, [len_train, len_val])

    # Wrap datasets with loaders.
    data_train = torch.utils.data.DataLoader(
        dataset=data_train,
        shuffle=True,
        batch_size=batch_size,
        num_workers=num_workers,
        drop_last=True)
    data_val = torch.utils.data.DataLoader(
        dataset=data_val,
        shuffle=True,
        batch_size=batch_size,
        num_workers=num_workers,
        drop_last=False)
    data_test = torch.utils.data.DataLoader(
        dataset=data_test,
        shuffle=False,
        batch_size=batch_size,
        num_workers=num_workers,
        drop_last=False)

    return data_train, data_val, data_test


def get_model():
    """Return a ImageNet-pretrained ResNet-34 model, resized.

    Returns:
        (torch.nn.Module): The model, resized for the target task.
    """
    # Load the pretrained model.
    model = torchvision.models.resnet34(pretrained=True)

    # Resize model for our task.
    model.conv1 = torch.nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1,
                                  bias=False)
    model.avgpool = torch.nn.AvgPool2d(2)

    return model


def load_checkpoint(checkpoint, model, optimizer=None):
    """Load state from the given checkpoint file.

    Args:
        checkpoint (str): The path of the checkpoint file.
        model (torch.nn.Module): The model to load state.
        optimizer (torch.optim.Optimizer, optional): The optimizer to load
               state.
    """
    model_state_dict, optimizer_state_dict = torch.load(checkpoint)
    model.load_state_dict(model_state_dict)

    if optimizer is not None:
        optimizer.load_state_dict(optimizer_state_dict)


def train(model, loss_function, optimizer, data):
    """Train the model on the given data.

    Args:
        model (torch.nn.Module): A PyTorch model.
        loss_function (torch.nn.Module): The loss function to compare model
            outputs with target values.
        optimizer (torch.optim.Optimizer): The optimizer algorithm to train the
            model.
        data (torch.utils.data.DataLoader): The data to train on.

    Returns:
        (float): The mean batch loss.
    """
    loss_sum = 0

    # Set the model in train mode.
    model.train()

    # Create progress bar.
    progress_bar = tqdm.tqdm(total=len(data),
                             unit='batch',
                             desc='[train] batch loss: 0.000',
                             leave=False)

    # Loop through training batches.
    for inputs, targets in data:

        # Reset gradients.
        optimizer.zero_grad()

        # Send data to GPU if CUDA is enabled.
        if next(model.parameters()).is_cuda:
            inputs = inputs.cuda()
            targets = targets.cuda()

        # Feed forward.
        with torch.set_grad_enabled(True):
            outputs = model(inputs)

        # Compute loss.
        loss = loss_function(outputs, targets)

        # Compute gradients.
        loss.backward()

        # Update parameters.
        optimizer.step()

        # Update progress bar.
        progress_bar.update(1)
        progress_bar.set_description(
            '[train] batch loss: {loss:.3f}'.format(loss=loss.item()))

        # Accumulate loss sum.
        loss_sum += loss.item()

    # Close progress bar.
    progress_bar.close()

    return loss_sum / len(data)


def evaluate(model, data):
    """Evaluate the model on the given data.

    Args:
        model (torch.nn.Module): A PyTorch model.
        data (torch.utils.data.DataLoader): The data to train on.

    Returns:
        (float): The overall accuracy.
    """
    n_targets = 0
    n_correct_predictions = 0

    # Set the model on evaluatio mode.
    model.eval()

    # Create progress bar.
    progress_bar = tqdm.tqdm(total=len(data),
                             unit='batch',
                             desc='[evaluate] batch accuracy: 0.000',
                             leave=False)

    # Loop through validation batches.
    for inputs, targets in data:

        # Send data to GPU if CUDA is enabled.
        if next(model.parameters()).is_cuda:
            inputs = inputs.cuda()
            targets = targets.cuda()

        # Feed forward.
        with torch.set_grad_enabled(False):
            outputs = model(inputs)

        # Choose the class with maximum probability.
        _, predictions = torch.max(outputs, 1)

        accuracy = (predictions == targets).sum().item() / len(targets)
        progress_bar.update(1)
        progress_bar.set_description(
            '[evaluate] batch accuracy: {accuracy:.3f}'.format(
                accuracy=accuracy))

        # Accumulate targets and correct predictions count.
        n_targets += len(targets)
        n_correct_predictions += (predictions == targets).sum().item()

    # Close progress bar.
    progress_bar.close()

    return n_correct_predictions / n_targets


def main():
    """Provide the main entrypoint.
    """
    # Fix random seed.
    torch.manual_seed(0)

    # Create checkpoint directory.
    try:
        os.mkdir('checkpoints')
    except FileExistsError:
        pass

    # Make preparations.
    args = get_args()
    logger = get_logger()
    data_train, data_val, data_test = get_data(args.batch_size,
                                               args.num_workers)
    model = get_model()

    # Log command arguments.
    logger.info(' '.join(sys.argv))
    logger.info(vars(args))

    # Send the model to the GPU, if enabled and available.
    if args.cuda:
        model = model.cuda()

    # Create the loss function and optimizer.
    loss_function = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(),
                                lr=args.learning_rate,
                                momentum=args.momentum)

    # Load checkpoint, if given.
    if args.checkpoint:
        load_checkpoint(args.checkpoint, model, optimizer)

    # Loop epochs.
    for epoch in range(args.num_epochs):
        logger.info(f'Epoch {epoch}:')

        mean_loss = train(model, loss_function, optimizer, data_train)
        logger.info(f'  - [training] mean loss: {mean_loss:.3f}')

        accuracy = evaluate(model, data_val)
        logger.info(f'  - [validation] accuracy: {accuracy:.3f}')

        torch.save([model.state_dict(), optimizer.state_dict()],
                   os.path.join('checkpoints', f'{epoch}.pth'))

    # Run final evaluation on the test data.
    logger.info('Test:')
    accuracy = evaluate(model, data_test)
    logger.info(f'  - [test] accuracy: {accuracy:.3f}')


if __name__ == '__main__':
    main()
