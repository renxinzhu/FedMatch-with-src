from datasets import generate_test_dataloader
import torch
from torch import nn
from torch.utils.data.dataloader import DataLoader
import copy
from typing import Any, Dict, OrderedDict
from logger import Logger

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

IStateDict = OrderedDict[str, torch.Tensor]
LR_DECAY_PATIENCE = 5

class Backbone(nn.Module):
    #def __init__(self, dataloader: DataLoader):
    #    super().__init__()

    def __init__(self, dataloader: Dict[str, DataLoader], device: torch.device,
                 sigma: IStateDict = None, phi: IStateDict = None, client_id: int = None, psi_factor=0.2):
        super().__init__()
        self.client_id = client_id
        self.logger = Logger(client_id=client_id)
        self.device = device
        self.dataloader = dataloader
        self.params = {
            "sigma": sigma if sigma else create_model().state_dict(),
            "phi": phi if phi else create_model().state_dict()
        }
        if phi is None:
            for key in self.params["phi"]:
                self.params["phi"][key] = self.params["sigma"][key] * psi_factor

        self.lr_decay_patience = LR_DECAY_PATIENCE
        self.lowest_vloss = float('inf')


        def conv_with_relu(*args, **kargs):
            return nn.Sequential(
                nn.Conv2d(*args, **kargs),
                nn.ReLU(),
            )

        self.max_pool = nn.MaxPool2d(2, 2)
        # self.last_max_pool = nn.MaxPool2d(4, 4)
        self.last_max_pool = nn.MaxPool2d(3, 1)
        self.flatten = nn.Flatten()

        self.cnn_block1 = nn.Sequential(
            conv_with_relu(3, 64, 3, padding=1),
            conv_with_relu(64, 128, 3, padding=1),
        )

        self.cnn_block2 = nn.Sequential(
            conv_with_relu(128, 128, 3, padding=1),
            conv_with_relu(128, 128, 3, padding=1),
        )

        self.cnn_block3 = conv_with_relu(128, 256, 3, padding=1)

        self.cnn_block4 = conv_with_relu(256, 512, 3, padding=1)

        self.cnn_block5 = nn.Sequential(
            conv_with_relu(512, 512, 3, padding=1),
            conv_with_relu(512, 512, 3, padding=1),
        )

        self.fcl = nn.Linear(512, 10)

        self.to(device)

        self.dataloader = dataloader

    def adjust_lr(self, vloss: float):
        if vloss < self.lowest_vloss:
            self.lr_decay_patience = LR_DECAY_PATIENCE
            self.lowest_vloss = vloss
            return

        self.lr_decay_patience -= 1
        if self.lr_decay_patience == 0:
            self.hyper_parameters['lr'] *= self.hyper_parameters['wd']
            self.lr_decay_patience = LR_DECAY_PATIENCE

    def forward(self, X):
        out = self.cnn_block1(X)
        out = self.max_pool(out)

        out = self.cnn_block2(out) + out
        out = self.cnn_block3(out)
        out = self.max_pool(out)

        out = self.cnn_block4(out)
        out = self.max_pool(out)

        out = self.cnn_block5(out) + out
        local_last_feature_map = self.last_max_pool(out)

        out = self.fcl(self.flatten(local_last_feature_map))

        return out,local_last_feature_map

    def replace_parameters(self, parameters):
        for idx, parameter in enumerate(self.parameters()):
            parameter.data = parameters[idx]

    def load_hyper_parameters(self, hyper_parameters: Dict[str, Any]):
        self.hyper_parameters = copy.deepcopy(hyper_parameters)

    def validate(self):
        correct = 0
        dataloader = generate_test_dataloader(1000)

        with torch.no_grad():
            for X, y in dataloader:
                X = X.to(device)
                y = y.to(device)

                pred = self.forward(X)
                correct += (pred.argmax(1) == y).type(torch.float).sum().item()

        return correct / len(dataloader.dataset)

    def evaluate(self, dataloader: DataLoader):
        correct = 0
        running_loss = 0
        model = NetDecomposed(
            freezed=self.params['sigma'], unfreezed=self.params['phi'])
        loss_fn = torch.nn.CrossEntropyLoss()
        model.eval()

        with torch.no_grad():
            for X, y in dataloader:
                pred = model(X)
                correct += (pred.argmax(1) == y).type(torch.float).sum().item()
                running_loss += loss_fn(pred, y).item() * X.size(0)

        test_acc = correct / len(dataloader.dataset)
        test_loss = running_loss / len(dataloader.dataset)

        return test_acc, test_loss


class NetDecomposed(nn.Module):
    def __init__(self, freezed: IStateDict, unfreezed: IStateDict):
        super().__init__()
        self.freezed = create_model()
        self.unfreezed = create_model()

        self.freezed.load_state_dict(freezed)
        self.unfreezed.load_state_dict(unfreezed)

    def forward(self, X):
        out = X
        for i in range(len(self.freezed)):
            if isinstance(self.freezed[i], (nn.Conv2d, nn.Linear)):
                out = self.freezed[i](out) + self.unfreezed[i](out)
            else:
                out = self.unfreezed[i](out)
        return out

    def get_freezed(self):
        return copy.deepcopy(self.freezed.state_dict())

    def get_unfreezed(self):
        return copy.deepcopy(self.unfreezed.state_dict())



def create_model():
    return nn.Sequential(nn.Conv2d(1, 32, 3, 1, padding=1),
                         nn.ReLU(),
                         nn.Conv2d(32, 64, 3, 1, padding=1),
                         nn.ReLU(),
                         nn.MaxPool2d(2, 2),

                         ###
                         nn.Sequential(
        nn.Conv2d(64, 64, 3, 1, padding=1),
        nn.ReLU(),
        nn.Conv2d(64, 64, 3, 1, padding=1),
        nn.ReLU()),

        nn.Conv2d(64, 128, 3, 1, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2, 2),
        nn.Conv2d(128, 256, 3, 1, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2, 2),

        ###
        nn.Sequential(
            nn.Conv2d(256, 256, 3, 1, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, 3, 1, padding=1),
            nn.ReLU()),

        nn.MaxPool2d(3, 1),

        nn.Flatten(),
        nn.Linear(256, 10),
    )