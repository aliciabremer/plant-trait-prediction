import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F 


class TransferDinoBig(nn.Module):
    def __init__(self, base_model, extra_features=163, dropout_features=0.2, dropout_after=0.2):
        super(TransferDinoBig, self).__init__()

        self.base_model = base_model

        # https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html#sphx-glr-beginner-transfer-learning-tutorial-py
        for param in self.base_model.parameters():
            param.requires_grad = False

        self.fc_extra1 = nn.Linear(extra_features, 256)
        self.fc_extra2 = nn.Linear(256, 512)
        self.fc_extra3 = nn.Linear(512, 512)
        self.dropout1 = nn.Dropout(dropout_features)
        
        self.fc_combined1 = nn.Linear(768+512, 2048)
        self.fc_combined2 = nn.Linear(2048, 2048)
        self.fc_combined3 = nn.Linear(2048, 6)
        
        self.dropout2 = nn.Dropout(dropout_after)
        self.dropout3 = nn.Dropout(dropout_after)

    def unfreeze_base_model_weights(self):
        # https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html#sphx-glr-beginner-transfer-learning-tutorial-py
        for param in self.base_model.parameters():
            param.requires_grad = True

    def train(self, mode=True):
        self.training = mode
        self.base_model.train(mode=mode)
        for module in self.children():
            module.train(mode)
        return self

    def forward(self, img, extra):
        res = self.base_model(img)

        ex = F.relu(self.fc_extra1(extra))
        ex = self.dropout1(F.relu(self.fc_extra2(ex)))
        ex = self.fc_extra3(ex)

        newx = torch.concat((res, ex),dim=1)
        
        x = self.dropout2(F.relu(self.fc_combined1(newx)))
        x = self.dropout3(F.relu(self.fc_combined2(x)))
        return self.fc_combined3(x)

class TransferSWINBatch(nn.Module):
    def __init__(self, base_model, extra_features=163, dropout_features=0.2, dropout_after=0.2):
        super(TransferSWINBatch, self).__init__()

        self.base_model = base_model

        # https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html#sphx-glr-beginner-transfer-learning-tutorial-py
        for param in self.base_model.parameters():
            param.requires_grad = False
        
        num_last_layer = self.base_model.head.in_features
        self.base_model.head = nn.Identity()

        self.fc_extra1 = nn.Linear(extra_features, 1024)
        self.fc_extra2 = nn.Linear(1024, 1024)
        
        self.dropout1 = nn.Dropout(dropout_features)
        self.batchnorm1 = nn.BatchNorm1d(1024)
        self.new_bn = nn.LayerNorm((1024,))
        
        self.fc_combined1 = nn.Linear(num_last_layer+1024, 1024)
        self.fc_combined2 = nn.Linear(1024, 512)
        self.fc_combined3 = nn.Linear(512, 256)
        self.fc_combined4 = nn.Linear(256, 6)
        
        self.dropout2 = nn.Dropout(dropout_after)
        self.dropout3 = nn.Dropout(dropout_after)
        self.dropout4 = nn.Dropout(dropout_after)
        
        self.batchnorm2 = nn.BatchNorm1d(1024)
        self.batchnorm3 = nn.BatchNorm1d(512)
        self.batchnorm4 = nn.BatchNorm1d(256)

    def unfreeze_base_model_weights(self):
        # https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html#sphx-glr-beginner-transfer-learning-tutorial-py
        for param in self.base_model.parameters():
            param.requires_grad = True

    def train(self, mode=True):
        self.training = mode
        self.base_model.train(mode=mode)
        for module in self.children():
            module.train(mode)
        return self
    
    def forward(self, img, extra):
        res = self.base_model(img)

        ex = self.batchnorm1(self.dropout1(F.relu(self.fc_extra1(extra))))
        ex = self.fc_extra2(ex)

        newx = torch.concat((res, ex),dim=1)
        
        x = self.batchnorm2(self.dropout2(F.relu(self.fc_combined1(newx))))
        x = self.batchnorm3(self.dropout3(F.relu(self.fc_combined2(x))))
        x = self.batchnorm4(F.relu(self.fc_combined3(x)))
        return self.fc_combined4(x)
