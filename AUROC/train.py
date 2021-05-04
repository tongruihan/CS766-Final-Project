from __future__ import division, print_function

import torch
import bisect
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
# import matplotlib.pyplot as plt
import loss as new_loss
import sklearn.metrics as skmetrics
from sklearn.model_selection import train_test_split
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader
from dataset import MaskDataset, SynDataset
from models import MobileNetV2


TRAINING_CONFIG = {
    'batch_size': 64,
    'shuffle': True,
    'num_workers': 12,
    'max_epoch': 100,
    'lr': 1,
    'lambda_lr': 0.001,
    'bias_lr': 0.001,
    'checkpoint_epoch': 1,
    'num_anchors': 20,
    'num_classes': 2,
    'rate_range': (0.0, 1.0),
    'mode': 'F@T',
    'save_plot': False
}

print(TRAINING_CONFIG)

def save_plot_roc(fpr, tpr):
    plt.plot(fpr, tpr, color='r')
    plt.ylabel('TPR')
    plt.xlabel('FPR')
    plt.ylim(top=1)
    plt.ylim(bottom=0)
    plt.xlim(right=1)
    plt.xlim(left=0)
    # plt.show()
    plt.savefig("ROC.jpg", dpi=600)


def get_fpr_tpr_for_thresh(fpr, tpr, thresh):
    p = bisect.bisect_left(tpr, thresh)
    fpr_thresh, tpr_thresh = fpr[p:], tpr[p:]
    if tpr[p] != thresh:
        # It indicates that tpr[p-1] < thresh < tpr[p]
        interpolate = fpr[p] - (tpr[p] - thresh) * (fpr[p] - fpr[p - 1]) / (tpr[p] - tpr[p - 1])
        fpr_thresh = np.insert(fpr_thresh, 0, interpolate)
        tpr_thresh = np.insert(tpr_thresh, 0, thresh)
    return fpr_thresh, tpr_thresh


def calculate_auc_at_tpr_beta(fpr, tpr, beta):
    fpr_thresh, tpr_thresh = get_fpr_tpr_for_thresh(fpr, tpr, beta)
    return skmetrics.auc(fpr_thresh, tpr_thresh) - (1 - fpr_thresh[0]) * tpr_thresh[0]


class Dataset(Dataset):

    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index], np.expand_dims(self.labels[index], axis=0)


def train(config, device):
    train_AUROC_list = list()
    test_AUROC_list = list()

    # model = torch.hub.load('pytorch/vision:v0.9.0', 'mobilenet_v2', pretrained=True)
    # model = MobileNetV2(num_classes=config['num_classes'])
    # model = model.to(device)
    model = torch.hub.load('pytorch/vision:v0.9.0', 'resnet18', pretrained=True)
    for param in model.parameters():
        param.requires_grad = False
    model.fc = nn.Sequential(nn.Linear(model.fc.in_features, 256),
                                nn.ReLU(),
                                nn.Dropout(0.2),
                                nn.Linear(256, config['num_classes'])
                            )
    model = model.to(device)

    # training_loader = DataLoader(MaskDataset(file_path="dataset/", train=True), batch_size=config["batch_size"], shuffle=config["shuffle"])
    # eval_loader = DataLoader(MaskDataset(file_path="dataset/", train=False), batch_size=config["batch_size"], shuffle=config["shuffle"])

    dataset = SynDataset(file_path="syn_dataset")
    labels = dataset.labels
    train_idx, valid_idx= train_test_split(
        np.arange(len(labels)),
        test_size=0.2,
        shuffle=True,
        stratify=labels
    )
    train_sampler = torch.utils.data.SubsetRandomSampler(train_idx)
    valid_sampler = torch.utils.data.SubsetRandomSampler(valid_idx)

    training_loader = torch.utils.data.DataLoader(dataset, batch_size=config["batch_size"], sampler=train_sampler)
    eval_loader = torch.utils.data.DataLoader(dataset, batch_size=config["batch_size"], sampler=valid_sampler)

    lambdas = torch.rand(config['num_classes'], config['num_anchors']).clamp(min=0)
    lambdas = lambdas / torch.norm(lambdas, 1)
    # lambdas = torch.zeros(config['num_classes'], config['num_anchors'])
    lambdas = nn.Parameter(lambdas.to(device))
    bias = torch.rand(config['num_classes'], config['num_anchors'])
    bias = nn.Parameter(bias.to(device))
    optimizer_model = optim.SGD(model.parameters(), lr=config['lr'], momentum=0.9)
    optimizer_lambdas = optim.SGD([lambdas], lr=config['lambda_lr'], momentum=0.9)
    optimizer_bias = optim.SGD([bias], lr=config['bias_lr'], momentum=0.9)

    criterion = new_loss.AUCROCLoss(config, device)

    for epoch in range(config['max_epoch']):
        model.train()
        running_loss = 0
        total_labels, total_preds = list(), list()
        eval_labels, eval_preds = list(), list()
        for batch_idx, (data, labels) in enumerate(training_loader):
            data, labels = data.float(), labels.float()
            labels = labels.unsqueeze(-1)
            data, labels = data.to(device), labels.to(device)

            optimizer_model.zero_grad()
            optimizer_bias.zero_grad()

            logits = model(data)
            probabilities = F.softmax(logits, dim=1)
            loss = criterion(labels, logits, probabilities, lambdas, bias, device)

            loss.backward()
            optimizer_model.step()
            optimizer_bias.step()
        # print("Before update lambdas: \n", lambdas[:, :config['num_anchors']])
        for batch_idx, (data, labels) in enumerate(training_loader):
            data, labels = data.float(), labels.float()
            labels = labels.unsqueeze(-1)
            data, labels = data.to(device), labels.to(device)

            optimizer_lambdas.zero_grad()

            logits = model(data)
            probabilities = F.softmax(logits, dim=1)
            loss = criterion(labels, logits, probabilities, lambdas, bias, device)

            loss.backward()
            optimizer_lambdas.step()

            lambdas.data = lambdas.clamp(min=0).data

            total_labels += list(labels.cpu().detach().numpy())
            total_preds += list(probabilities[:, 1].cpu().detach().numpy())

            running_loss += loss
        # print("After update lambdas: \n", lambdas[:, :config['num_anchors']])
        if config['save_plot']:
            fpr, tpr, thresholds = skmetrics.roc_curve(np.asarray(total_labels), np.asarray(total_preds), pos_label=1)
            save_plot_roc(fpr, tpr)
        train_roc_auc = skmetrics.roc_auc_score(np.asarray(total_labels), np.asarray(total_preds))
        fpr, tpr, thresholds = skmetrics.roc_curve(np.asarray(total_labels), np.asarray(total_preds), pos_label=1)
        partial_train_roc_auc = calculate_auc_at_tpr_beta(fpr, tpr, beta=config['rate_range'][0])
        train_AUROC_list.append(train_roc_auc)
        print("Train - epoch {} ROC_AUC_Loss = {:.3f}, AUROC = {:.4f}, partial_AUROC = {:.6f}".format(
            epoch + 1,
            running_loss,
            train_roc_auc,
            partial_train_roc_auc))

        if epoch == 0 or (epoch + 1) % config['checkpoint_epoch'] == 0:
            model.eval()
            valid_running_loss = 0
            with torch.no_grad():
                for _, (data, labels) in enumerate(eval_loader):
                    data, labels = data.float(), labels.float()
                    data, labels = data.to(device), labels.to(device)
                    labels = labels.unsqueeze(-1)

                    logits = model(data)
                    probabilities = F.softmax(logits, dim=1)
                    loss = criterion(labels, logits, probabilities, lambdas, bias, device)

                    eval_labels += list(labels.cpu().detach().numpy())
                    eval_preds += list(probabilities[:, 1].cpu().detach().numpy())

                    valid_running_loss += loss

            eval_roc_auc = skmetrics.roc_auc_score(np.asarray(eval_labels), np.asarray(eval_preds))
            fpr, tpr, thresholds = skmetrics.roc_curve(np.asarray(eval_labels), np.asarray(eval_preds), pos_label=1)
            partial_eval_roc_auc = calculate_auc_at_tpr_beta(fpr, tpr, beta=config['rate_range'][0])
            test_AUROC_list.append(eval_roc_auc)
            print("Eval  - epoch {} ROC_AUC_Loss = {:.3f}, AUROC = {:.4f}, partial_AUROC = {:.6f}".format(
                epoch + 1,
                valid_running_loss,
                eval_roc_auc,
                partial_eval_roc_auc))

    return np.array(train_AUROC_list), np.array(test_AUROC_list)


def classifier_train(config, device):
    train_AUROC_list = list()
    test_AUROC_list = list()

    model = torch.hub.load('pytorch/vision:v0.9.0', 'resnet18', pretrained=True)
    for param in model.parameters():
        param.requires_grad = False
    model.fc = nn.Sequential(nn.Linear(model.fc.in_features, 256),
                                nn.ReLU(),
                                nn.Dropout(0.2),
                                nn.Linear(256, config['num_classes'])
                            )
    model = model.to(device)

    # training_loader = DataLoader(MaskDataset(file_path="dataset/", train=True), batch_size=config["batch_size"], shuffle=config["shuffle"])
    # eval_loader = DataLoader(MaskDataset(file_path="dataset/", train=False), batch_size=config["batch_size"], shuffle=config["shuffle"])

    dataset = SynDataset(file_path="syn_dataset")
    labels = dataset.labels
    train_idx, valid_idx= train_test_split(
        np.arange(len(labels)),
        test_size=0.2,
        shuffle=True,
        stratify=labels
    )
    train_sampler = torch.utils.data.SubsetRandomSampler(train_idx)
    valid_sampler = torch.utils.data.SubsetRandomSampler(valid_idx)

    training_loader = torch.utils.data.DataLoader(dataset, batch_size=config["batch_size"], sampler=train_sampler)
    eval_loader = torch.utils.data.DataLoader(dataset, batch_size=config["batch_size"], sampler=valid_sampler)

    optimizer_model = optim.Adam(model.fc.parameters(), lr=config['lr'])

    criterion = nn.CrossEntropyLoss()

    for epoch in range(config['max_epoch']):
        model.train()
        running_loss = 0
        total_labels, total_preds = list(), list()
        eval_labels, eval_preds = list(), list()
        for batch_idx, (data, labels) in enumerate(training_loader):
            data, labels = data.float(), labels.long()
            data, labels = data.to(device), labels.to(device)

            optimizer_model.zero_grad()
            logits = model(data)
            probabilities = F.softmax(logits, dim=1)
            loss = criterion(logits, labels)

            loss.backward()
            optimizer_model.step()

            total_labels += list(labels.cpu().detach().numpy())
            total_preds += list(probabilities[:, 1].cpu().detach().numpy())

            running_loss += loss
        # print("After update lambdas: \n", lambdas[:, :config['num_anchors']])
        if config['save_plot']:
            fpr, tpr, thresholds = skmetrics.roc_curve(np.asarray(total_labels), np.asarray(total_preds), pos_label=1)
            save_plot_roc(fpr, tpr)
        train_roc_auc = skmetrics.roc_auc_score(np.asarray(total_labels), np.asarray(total_preds))
        fpr, tpr, thresholds = skmetrics.roc_curve(np.asarray(total_labels), np.asarray(total_preds), pos_label=1)
        partial_train_roc_auc = calculate_auc_at_tpr_beta(fpr, tpr, beta=config['rate_range'][0])
        train_AUROC_list.append(train_roc_auc)
        print("Train - epoch {} ROC_AUC_Loss = {:.3f}, AUROC = {:.4f}, partial_AUROC = {:.6f}".format(
            epoch + 1,
            running_loss,
            train_roc_auc,
            partial_train_roc_auc))

        if epoch == 0 or (epoch + 1) % config['checkpoint_epoch'] == 0:
            model.eval()
            valid_running_loss = 0
            with torch.no_grad():
                for _, (data, labels) in enumerate(eval_loader):
                    data, labels = data.float(), labels.long()
                    data, labels = data.to(device), labels.to(device)

                    logits = model(data)
                    probabilities = F.softmax(logits, dim=1)
                    loss = criterion(logits, labels)

                    eval_labels += list(labels.cpu().detach().numpy())
                    eval_preds += list(probabilities[:, 1].cpu().detach().numpy())

                    valid_running_loss += loss

            eval_roc_auc = skmetrics.roc_auc_score(np.asarray(eval_labels), np.asarray(eval_preds))
            fpr, tpr, thresholds = skmetrics.roc_curve(np.asarray(eval_labels), np.asarray(eval_preds), pos_label=1)
            partial_eval_roc_auc = calculate_auc_at_tpr_beta(fpr, tpr, beta=config['rate_range'][0])
            test_AUROC_list.append(eval_roc_auc)
            print("Eval  - epoch {} ROC_AUC_Loss = {:.3f}, AUROC = {:.4f}, partial_AUROC = {:.6f}".format(
                epoch + 1,
                valid_running_loss,
                eval_roc_auc,
                partial_eval_roc_auc))

    return np.array(train_AUROC_list), np.array(test_AUROC_list)


if __name__ == "__main__":
    device = torch.device("cuda:5" if torch.cuda.is_available() else "cpu")
    print(device)

    train_auroc, test_auroc = classifier_train(TRAINING_CONFIG, device)
