import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.nn import Parameter


class CifarCNN(nn.Module):
    def __init__(self, num_classes=10):
        super(CifarCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=5, padding=0)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, 5, padding=1)
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
        self.fc1 = nn.Linear(64 * 3 * 3, 128)
        self.fc2 = nn.Linear(128, num_classes, bias=True)
        self.base_weight_keys = [
            'conv1.weight', 'conv1.bias',
            'conv2.weight', 'conv2.bias',
            'conv3.weight', 'conv3.bias',
            'fc1.weight', 'fc1.bias',
        ]
        self.classifier_weight_keys = [
            'fc2.weight', 'fc2.bias',
        ]

    def forward(self, x):
        x = self.pool(F.leaky_relu(self.conv1(x)))
        x = self.pool(F.leaky_relu(self.conv2(x)))
        x = self.pool(F.leaky_relu(self.conv3(x)))
        x = x.view(-1, self.num_flat_features(x))
        x = F.leaky_relu(self.fc1(x))
        y = self.fc2(x)
        return x, y

    def feature2logit(self, x):
        return self.fc2(x)

    def num_flat_features(self, x):
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


class CNN_FMNIST(nn.Module):
    def __init__(self, num_classes=10):
        super(CNN_FMNIST, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=5, padding=0)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, 5, padding=1)
        self.fc1 = nn.Linear(32 * 5 * 5, 128)
        self.fc2 = nn.Linear(128, num_classes, bias=True)
        self.base_weight_keys = ['conv1.weight', 'conv1.bias',
                                 'conv2.weight', 'conv2.bias',
                                 'fc1.weight', 'fc1.bias', ]
        self.classifier_weight_keys = ['fc2.weight', 'fc2.bias', ]

    def forward(self, x):
        x = self.pool(F.leaky_relu(self.conv1(x)))
        x = self.pool(F.leaky_relu(self.conv2(x)))
        x = x.view(-1, self.num_flat_features(x))
        x = F.leaky_relu(self.fc1(x))
        y = self.fc2(x)
        return x, y

    def feature2logit(self, x):
        return self.fc2(x)

    def num_flat_features(self, x):
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


class Hypernetwork_CifarCNN(nn.Module):
    def __init__(self, embedding_dim, hidden_dim):
        super(Hypernetwork_CifarCNN, self).__init__()

        layers = [nn.Linear(embedding_dim, hidden_dim),
                  nn.ReLU(inplace=True)]
        self.mlp = nn.Sequential(*layers)

        # generate feature extractor parameter
        self.c1_weights = nn.Linear(hidden_dim, 16 * 3 * 5 * 5)
        self.c1_bias = nn.Linear(hidden_dim, 16)
        self.c2_weights = nn.Linear(hidden_dim, 32 * 16 * 5 * 5)
        self.c2_bias = nn.Linear(hidden_dim, 32)
        self.c3_weights = nn.Linear(hidden_dim, 64 * 32 * 3 * 3)
        self.c3_bias = nn.Linear(hidden_dim, 64)
        self.l1_weights = nn.Linear(hidden_dim, 128 * 64 * 3 * 3)
        self.l1_bias = nn.Linear(hidden_dim, 128)

    def forward(self, client_embedding):
        features = self.mlp(client_embedding)
        weights = {
            "conv1.weight": self.c1_weights(features).view(16, 3, 5, 5),
            "conv1.bias": self.c1_bias(features).view(-1),
            "conv2.weight": self.c2_weights(features).view(32, 16, 5, 5),
            "conv2.bias": self.c2_bias(features).view(-1),
            "conv3.weight": self.c3_weights(features).view(64, 32, 3, 3),
            "conv3.bias": self.c3_bias(features).view(-1),
            "fc1.weight": self.l1_weights(features).view(128, 64 * 3 * 3),
            "fc1.bias": self.l1_bias(features).view(-1),
        }
        return weights


class Hypernetwork_CNN_FMNIST(nn.Module):
    def __init__(self, embedding_dim, hidden_dim):
        super(Hypernetwork_CNN_FMNIST, self).__init__()

        layers = [nn.Linear(embedding_dim, hidden_dim),
                  nn.ReLU(inplace=True)]
        self.mlp = nn.Sequential(*layers)

        # generate feature extractor parameter
        self.c1_weights = nn.Linear(hidden_dim, 16 * 1 * 5 * 5)
        self.c1_bias = nn.Linear(hidden_dim, 16)
        self.c2_weights = nn.Linear(hidden_dim, 32 * 16 * 5 * 5)
        self.c2_bias = nn.Linear(hidden_dim, 32)
        self.l1_weights = nn.Linear(hidden_dim, 128 * 32 * 5 * 5)
        self.l1_bias = nn.Linear(hidden_dim, 128)

    def forward(self, client_embedding):
        features = self.mlp(client_embedding)
        weights = {
            "conv1.weight": self.c1_weights(features).view(16, 1, 5, 5),
            "conv1.bias": self.c1_bias(features).view(-1),
            "conv2.weight": self.c2_weights(features).view(32, 16, 5, 5),
            "conv2.bias": self.c2_bias(features).view(-1),
            "fc1.weight": self.l1_weights(features).view(128, 32 * 5 * 5),
            "fc1.bias": self.l1_bias(features).view(-1),
        }
        return weights


class CifarCNN_Hyper(nn.Module):
    def __init__(self, args):
        super(CifarCNN_Hyper, self).__init__()
        self.target_model = CifarCNN(num_classes=args.num_classes)  # original model
        self.classifier_weight_keys = [
            'target_model.fc2.weight', 'target_model.fc2.bias',
        ]

        # client embedding
        self.client_embedding = nn.Embedding(num_embeddings=1, embedding_dim=args.embed_dim)
        # hypernetwork
        self.hypernetwork = Hypernetwork_CifarCNN(embedding_dim=args.embed_dim, hidden_dim=args.hidden_dim)

    def forward(self, x):
        # generate feature extractor weight
        client_embedding = self.client_embedding(torch.tensor(0).cuda())
        weights = self.hypernetwork(client_embedding)
        self.target_model.load_state_dict(weights, strict=False)  # only load feature extractor weights

        x, y = self.target_model(x)
        return x, y

    def predict(self, x):
        x, y = self.target_model(x)
        return y

    def generate_weight(self):
        client_embedding = self.client_embedding(torch.tensor(0).cuda())
        weights = self.hypernetwork(client_embedding)
        return weights


class CNN_FMNIST_Hyper(CifarCNN_Hyper):
    def __init__(self, args):
        super(CNN_FMNIST_Hyper, self).__init__(args)
        self.target_model = CNN_FMNIST(num_classes=args.num_classes)  # original model

        # hypernetwork
        self.hypernetwork = Hypernetwork_CNN_FMNIST(embedding_dim=args.embed_dim, hidden_dim=args.hidden_dim)
