from flwr.common import Context, ndarrays_to_parameters
from flwr.server import ServerApp, ServerAppComponents, ServerConfig
from flwr.client import ClientApp
from flwr.server.strategy import FedAvg
from cycle_consistent_federated_learning.task import Net, get_weights
from cycle_consistent_federated_learning.cycle_cons_strategy import CycleConsistentAvg
from cycle_consistent_federated_learning.server_app import server_fn
from cycle_consistent_federated_learning.client_app import client_fn
import torch
from flwr.simulation import run_simulation

server_app = ServerApp(server_fn=server_fn)
client_app = ClientApp(client_fn=client_fn)

backend_config = {"client_resources": {"num_gpus": 0.1, "num_cpus": 1}}


run_simulation(
    server_app=server_app,
    client_app=client_app,
    backend_config=backend_config,
    num_supernodes=10,
)
