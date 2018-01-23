#!/usr/bin/env python3
"""Define a convenience wrapper class for PyTorch models.
"""
import numpy
import torch
import torch.autograd

class Model:
    """Wraps a PyTorch model to provide convenience functions.

    Args:
        model (torch.nn.Module): The PyTorch model.
        loss_function (torch.nn.Module): The loss function.
        optimizer (torch.optim.Optimzer): The training optimizer.
    """

    def __init__(self, model, loss_function, optimizer):
        self.model = model
        self.loss_function = loss_function
        self.optimizer = optimizer

        # Send model to GPU if possible.
        self._use_cuda = torch.cuda.is_available()
        if self._use_cuda:
            self.model = self.model.cuda()

    def train_batch(self, x, y):
        """Train a batch.

        Args:
            x (torch.Tensor): The input data.
            y (torch.Tensor): The target label.

        Returns:
            float: The batch loss.
        """
        # Put the model in training mode.
        self.model.train()

        # Send tensors to GPU if possible.
        if self._use_cuda:
            x = x.cuda()
            y = y.cuda()

        self.optimizer.zero_grad()

        # Forward pass.
        x = torch.autograd.Variable(x, requires_grad=True)
        y_output = self.model(x)

        # Backward pass.
        y = torch.autograd.Variable(y)
        loss = self.loss_function(y_output, y)
        loss.backward()

        self.optimizer.step()

        return loss.cpu().data

    def infer_batch(self, x):
        """Infer a batch.

        Args:
            x (torch.Tensor): The input data.

        Returns:
            torch.Tensor: The inferred labels.
        """
        # Put the model in evaluation mode.
        self.model.eval()

        # Send tensor to GPU if possible.
        if self._use_cuda:
            x = x.cuda()

        # Forward pass.
        x = torch.autograd.Variable(x, volatile=True)
        y_output = self.model(x)

        # Get indices of the largest class.
        _, y_pred = torch.max(y_output.cpu().data, 1)

        return y_pred

    def load(self, path):
        """Load model and optimizer parameters from a saved checkpoint file.

        Args:
            path (str): The path of the saved checkpoint file.
        """
        checkpoint = torch.load(path)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    def save(self, path):
        """Save model and optimizer parameters to a checkpoint file.

        Args:
            path (str): The path to save the checkpoint file to.
        """
        checkpoint = {'model_state_dict': self.model.state_dict(),
                      'optimizer_state_dict': self.optimizer.state_dict()}

        torch.save(checkpoint, path)
