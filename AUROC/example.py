from __future__ import division
from __future__ import print_function

import torch
import bisect
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
import loss as new_loss
import sklearn.metrics as skmetrics
from torch.utils.data import Dataset, DataLoader


EXPERIMENT_DATA_CONFIG = {
    'positives_centers': [[0, 1.0], [1, -0.5]],
    'negatives_centers': [[0, -0.5], [1, 1.0]],
    'positives_variances': [0.15, 0.1],
    'negatives_variances': [0.15, 0.1],
    'positives_counts': [500, 50],
    'negatives_counts': [2000, 100]
}

TRAINING_CONFIG = {
    'batch_size': 64,
    'shuffle': True,
    'num_workers': 6,
    'max_epoch': 40,
    'lr': 0.001,
    'checkpoint_epoch': 5,
    'num_anchors': 20,
    'num_classes': 2,
    'class_priors': 550/3100,
    'rate_range': (0.0, 1.0),
    'mode': 'F@T',
    'save_plot': False
}


def create_data_for_experiment(**data_config):
    """
    Synthesize a binary-labeled dataset whose data is a mixture of four Gaussians - two positives and two negatives.
    The centers and variances are defined in the respective keys of data_config.
    :param data_config: Configurations for data synthesis.
    :return: A dictionary with two shuffled datasets - one for training and one for evaluation. The datapoints are
    two-dimensional floats, and the labels are in {0, 1}.
    """
    def data_points(is_positives, index):
        """
        Creates a set of data which follows the normal distribution with mean and variance specified in config
        :param is_positives: class label, 0 - positive, 1 - negative
        :param index: from which distribution
        :return: a set of data
        """
        variance = data_config['positives_variances' if is_positives else 'negatives_variances'][index]
        center = data_config['positives_centers' if is_positives else 'negatives_centers'][index]
        count = data_config['positives_counts' if is_positives else 'negatives_counts'][index]

        return variance * np.random.randn(count, 2) + np.array([center])

    def create_data():
        return np.concatenate([data_points(False, 0),
                               data_points(True, 0),
                               data_points(True, 1),
                               data_points(False, 1)], axis=0)

    def create_label():
        return np.array([0.0] * data_config['negatives_counts'][0] +
                        [1.0] * data_config['positives_counts'][0] +
                        [1.0] * data_config['positives_counts'][1] +
                        [0.0] * data_config['negatives_counts'][1])

    permutation = np.random.permutation(sum(data_config["positives_counts"] + data_config["negatives_counts"]))
    train_data = create_data()[permutation, :]
    eval_data = create_data()[permutation, :]
    train_labels = create_label()[permutation]
    eval_labels = create_label()[permutation]

    return {
        'train_data': train_data,
        'train_labels': train_labels,
        'eval_data': eval_data,
        'eval_labels': eval_labels
    }


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
        interpolate = fpr[p] - (tpr[p] - thresh) * (fpr[p] - fpr[p-1]) / (tpr[p] - tpr[p-1])
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


class OneLayerModel(nn.Module):

    def __init__(self):
        super(OneLayerModel, self).__init__()
        self.fc = nn.Linear(in_features=2, out_features=2, bias=True)
        self.fc.weight.data.fill_(-1.0)
        self.fc.bias.data.fill_(0)

    def forward(self, x):
        return self.fc(x)


class TwoLayerModel(nn.Module):

    def __init__(self):
        super(TwoLayerModel, self).__init__()
        self.fc1 = nn.Linear(in_features=2, out_features=4, bias=True)
        self.fc2 = nn.Linear(in_features=4, out_features=2, bias=True)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return self.fc2(x)


def train(model, config):
    sample_data = create_data_for_experiment(**EXPERIMENT_DATA_CONFIG)
    # positive = (sample_data['train_labels'] == 1)
    # negative = (sample_data['train_labels'] == 0)
    # plt.scatter(sample_data['train_data'][positive][:, 0], sample_data['train_data'][positive][:, 1], c='blue')
    # plt.scatter(sample_data['train_data'][negative][:, 0], sample_data['train_data'][negative][:, 1], c='red')
    # plt.show()
    training_set = Dataset(sample_data['train_data'], sample_data['train_labels'])
    training_loader = DataLoader(training_set,
                                 batch_size=config['batch_size'],
                                 shuffle=config['shuffle'],
                                 num_workers=config['num_workers'])

    eval_set = Dataset(sample_data['eval_data'], sample_data['eval_labels'])
    eval_loader = DataLoader(eval_set,
                             batch_size=config['batch_size'],
                             shuffle=config['shuffle'],
                             num_workers=config['num_workers'])

    lambdas = nn.Parameter(torch.rand(config['num_classes'], config['num_anchors']))
    bias = nn.Parameter(torch.zeros(config['num_classes'], config['num_anchors']))
    optimizer_model = optim.SGD(model.parameters(), lr=config['lr'], momentum=0.9)
    optimizer_lambda = optim.SGD([lambdas], lr=config['lr'], momentum=0.9)
    optimizer_bias = optim.SGD([bias], lr=config['lr'], momentum=0.9)

    criterion = new_loss.AUCROCLoss(config)
    criterion = getattr(new_loss, 'AUCROCLoss')(config)

    for epoch in range(config['max_epoch']):
        model.train()
        running_loss = 0
        total_labels, total_preds = list(), list()
        eval_labels, eval_preds = list(), list()
        for batch_idx, (data, labels) in enumerate(training_loader):
            data, labels = data.float(), labels.float()
            optimizer_model.zero_grad()
            # optimizer_lambda.zero_grad()
            optimizer_bias.zero_grad()
            logits = model(data)
            probabilities = F.softmax(logits, dim=1)
            loss = criterion(labels, logits, lambdas, bias)
            loss.backward()
            optimizer_model.step()
            # optimizer_lambda.step()
            optimizer_bias.step()

            total_labels += list(labels.detach().numpy())
            total_preds += list(probabilities[:, 1].detach().numpy())

            running_loss += loss

        if config['save_plot']:
            fpr, tpr, thresholds = skmetrics.roc_curve(np.asarray(total_labels), np.asarray(total_preds), pos_label=1)
            save_plot_roc(fpr, tpr)
        train_roc_auc = skmetrics.roc_auc_score(np.asarray(total_labels), np.asarray(total_preds))
        # train_roc_auc = skmetrics.average_precision_score(np.asarray(total_labels),
        #                                                   np.asarray(total_preds),
        #                                                   pos_label=1)
        fpr, tpr, thresholds = skmetrics.roc_curve(np.asarray(total_labels), np.asarray(total_preds), pos_label=1)
        partial_train_roc_auc = calculate_auc_at_tpr_beta(fpr, tpr, beta=config['rate_range'][0])
        print("Train - epoch {} ROC_AUC_Loss = {:.3f}, AUROC = {:.4f}, partial_AUROC = {:.6f}".format(
            epoch + 1,
            running_loss,
            train_roc_auc,
            partial_train_roc_auc))
        # print(lambdas)

        if epoch == 0 or (epoch + 1) % config['checkpoint_epoch'] == 0:
            model.eval()
            valid_running_loss = 0
            with torch.no_grad():
                for _, (data, labels) in enumerate(eval_loader):
                    data, labels = data.float(), labels.float()
                    logits = model(data)
                    probabilities = F.softmax(logits, dim=1)
                    loss = criterion(labels, logits, lambdas, bias)

                    eval_labels += list(labels.detach().numpy())
                    eval_preds += list(probabilities[:, 1].detach().numpy())

                    valid_running_loss += loss

            eval_roc_auc = skmetrics.roc_auc_score(np.asarray(eval_labels), np.asarray(eval_preds))
            # eval_roc_auc = skmetrics.average_precision_score(np.asarray(eval_labels),
            #                                                  np.asarray(eval_preds),
            #                                                  pos_label=1)
            fpr, tpr, thresholds = skmetrics.roc_curve(np.asarray(total_labels), np.asarray(total_preds), pos_label=1)
            partial_eval_roc_auc = calculate_auc_at_tpr_beta(fpr, tpr, beta=config['rate_range'][0])
            print("Eval  - epoch {} ROC_AUC_Loss = {:.3f}, AUROC = {:.4f}, partial_AUROC = {:.6f}".format(
                epoch + 1,
                valid_running_loss,
                eval_roc_auc,
                partial_eval_roc_auc))


if __name__ == "__main__":
    # model = OneLayerModel()
    model = TwoLayerModel()
    train(model, TRAINING_CONFIG)
