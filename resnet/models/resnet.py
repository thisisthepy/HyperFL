import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.nn import Parameter
import timm
import os
from .models_resnet import ResNet18Fc, ResNet18Fc_Adapter


class ModelResNet(nn.Module):
    def __init__(self, num_classes=3):
        super(ModelResNet, self).__init__()
        self.feature_extractor = ResNet18Fc()
        # fixed base model
        for param in self.feature_extractor.parameters():
            param.requires_grad = False

        self.hidden_layer = nn.Linear(self.feature_extractor.output_num(), 128)
        self.fc = nn.Linear(128, num_classes)
        self.classifier_weight_keys = ['fc.weight', 'fc.bias']

    def forward(self, x):
        x = self.feature_extractor.forward(x)
        x = self.hidden_layer(x)
        y = self.fc(x)
        return x, y


class AdapterHyperNet(nn.Module):
    def __init__(self, embedding_dim, hidden_dim):
        super(AdapterHyperNet, self).__init__()

        layers = [nn.Linear(embedding_dim, hidden_dim),
                  nn.ReLU(inplace=True)]
        self.mlp = nn.Sequential(*layers)

        self.planes = [64, 128, 256, 512]

        self.adapters = nn.ModuleList([nn.ModuleList([nn.Linear(hidden_dim, plane),
                                                      nn.Linear(hidden_dim, plane),
                                                      nn.Linear(hidden_dim, plane * plane),
                                                      nn.Linear(hidden_dim, plane),
                                                      nn.Linear(hidden_dim, plane)
                                                      ])
                                       for plane in self.planes])

    def forward(self, client_embedding):
        features = self.mlp(client_embedding)
        weights = {}
        for i in range(4):
            keys = 'adapter.' + str(i)
            weights[keys + "conv.0.weight"] = self.adapters[i][0](features).view(-1)
            weights[keys + "conv.0.bias"] = self.adapters[i][1](features).view(-1)
            weights[keys + "conv.1.weight"] = self.adapters[i][2](features).view(self.planes[i], self.planes[i], 1, 1)
            weights[keys + "bn.weight"] = self.adapters[i][3](features).view(-1)
            weights[keys + "bn.bias"] = self.adapters[i][4](features).view(-1)
        return weights


class ModelResNet_Hyper(nn.Module):
    def __init__(self, args):
        super(ModelResNet_Hyper, self).__init__()
        self.feature_extractor = ResNet18Fc_Adapter()  # feature_extractor

        for name, param in self.feature_extractor.named_parameters():
            if 'adapter' not in name:
                param.requires_grad = False

        self.hidden_layer = nn.Linear(self.feature_extractor.output_num(), 128)
        self.fc = nn.Linear(128, args.num_classes)
        self.classifier_weight_keys = ['fc.weight', 'fc.bias']

        # client embedding
        self.client_embedding = nn.Embedding(num_embeddings=1, embedding_dim=args.embed_dim)
        # hypernetwork
        self.hypernetwork = AdapterHyperNet(embedding_dim=args.embed_dim, hidden_dim=args.hidden_dim)

        # for name, param in self.feature_extractor.named_parameters():
        #     print(name, param.shape)

    def forward(self, x):
        # generate weights
        client_embedding = self.client_embedding(torch.tensor(0).cuda())
        weights = self.hypernetwork(client_embedding)
        self.feature_extractor.load_state_dict(weights, strict=False)  # only load adapter weights

        x = self.feature_extractor.forward(x)
        x = self.hidden_layer(x)
        y = self.fc(x)
        return x, y

    def predict(self, x):
        x = self.feature_extractor.forward(x)
        x = self.hidden_layer(x)
        y = self.fc(x)
        return y

    def generate_weight(self):
        client_embedding = self.client_embedding(torch.tensor(0).cuda())
        weights = self.hypernetwork(client_embedding)
        return weights


class ModelResNet_Adapter(nn.Module):
    def __init__(self, args):
        super(ModelResNet_Adapter, self).__init__()
        self.feature_extractor = ResNet18Fc_Adapter()  # feature_extractor

        for name, param in self.feature_extractor.named_parameters():
            if 'adapter' not in name:
                param.requires_grad = False

        self.hidden_layer = nn.Linear(self.feature_extractor.output_num(), 128)
        self.fc = nn.Linear(128, args.num_classes)
        self.classifier_weight_keys = ['fc.weight', 'fc.bias']

    def forward(self, x):
        x = self.feature_extractor.forward(x)
        x = self.hidden_layer(x)
        y = self.fc(x)
        return x, y

    def predict(self, x):
        x = self.feature_extractor.forward(x)
        x = self.hidden_layer(x)
        y = self.fc(x)
        return y
