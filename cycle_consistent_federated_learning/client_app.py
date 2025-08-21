"""cycle-consistent-federated-learning: A Flower / PyTorch app."""

from flwr.client import NumPyClient, ClientApp
from flwr.common import Context

from cycle_consistent_federated_learning.task import (
    Net,
    DEVICE,
    load_data,
    get_weights,
    set_weights,
    train,
    test,
)

import torch
import numpy as np


# Define Flower Client and client_fn
class FlowerClient(NumPyClient):
    def __init__(self, net, trainloader, valloader, local_epochs):
        self.net = net
        self.trainloader = trainloader
        self.valloader = valloader
        self.local_epochs = local_epochs

    def fit(self, parameters, config):

        if config["reinit"]:
            print("Reinitializing weights and biases")
            parameters = reinitialize_ndarrays(parameters)
            set_weights(self.net, parameters)
        else:
            set_weights(self.net, parameters)

        results = train(
            self.net,
            self.trainloader,
            self.valloader,
            self.local_epochs,
            DEVICE,
        )
        return get_weights(self.net), len(self.trainloader.dataset), results

    def evaluate(self, parameters, config):
        set_weights(self.net, parameters)
        loss, accuracy = test(self.net, self.valloader)
        return loss, len(self.valloader.dataset), {"accuracy": accuracy}


def client_fn(context: Context):
    # Load model and data
    net = Net().to(DEVICE)
    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]
    trainloader, valloader = load_data(partition_id, num_partitions)
    local_epochs = context.run_config["local-epochs"]

    # Return Client instance
    return FlowerClient(net, trainloader, valloader, local_epochs).to_client()


# Flower ClientApp
app = ClientApp(
    client_fn,
)


def reinitialize_ndarrays(ndarrays):
    """
    Creates a list of new numpy ndarrays with the same shape as the input ndarrays,
    but with different random initialization suitable for neural networks.

    Args:
        ndarrays: A list of numpy ndarrays representing weights and biases of a neural network.

    Returns:
        A list of new numpy ndarrays with the same shape as the input ndarrays,
        but with different random initialization suitable for neural networks.
    """

    new_ndarrays = []
    for ndarray in ndarrays:
        # He initialization (good for ReLU-based networks)
        if len(ndarray.shape) > 1:  # For weights (matrices)
            fan_in = ndarray.shape[1]  # Number of input features
            stddev = np.sqrt(2 / fan_in)
            new_ndarray = np.random.randn(*ndarray.shape) * stddev
        else:  # For biases (vectors)
            new_ndarray = np.zeros_like(ndarray)  # Initialize biases to zero

        new_ndarrays.append(new_ndarray)

    return new_ndarrays
