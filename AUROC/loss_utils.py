import torch
import numpy as np


def range_to_anchors_and_delta(rate_range, num_anchors, device):
    rate_values = np.linspace(
        start=rate_range[0], stop=rate_range[1], num=num_anchors + 1)[1:]
    delta = (rate_range[1] - rate_range[0]) / num_anchors

    return torch.FloatTensor(rate_values).to(device), delta
        

def prepare_labels_weights(logits, labels, device, weights=None):
    batch_size, num_classes = logits.size()

    if torch.cuda.is_available():
        new_labels = torch.cuda.FloatTensor(batch_size, num_classes, device=device).zero_().scatter(1, labels.long().data, 1)
    else:
        new_labels = torch.zeros(batch_size, num_classes).scatter(1, labels.long().data, 1)
    
    if weights is None:
        if torch.cuda.is_available():
            weights = torch.cuda.FloatTensor(batch_size, device=device).data.fill_(1.0)
        else:
            weights = torch.ones(batch_size)

    if weights.dim() == 1:
        weights = weights.unsqueeze(-1)

    return new_labels, weights


def build_class_priors(labels,
                       device,
                       class_priors=None,
                       weights=None,
                       positive_pseudocount=1.0,
                       negative_pseudocount=1.0):
    if class_priors is not None:
        return class_priors

    if weights is None:
        # if torch.cuda.is_available():
        #     weights = torch.FloatTensor(labels.size(0)).data.fill_(1.0).to(device)
        # else:
            weights = torch.ones(labels.size(0))
            weights = weights.to(device)

    weighted_positives = (weights.unsqueeze(1) * labels).sum(0)
    weighted_sum = weights.sum(0)

    class_priors = torch.div(weighted_positives + positive_pseudocount,
                             weighted_sum + positive_pseudocount + negative_pseudocount)

    return class_priors


def weighted_hinge_loss(labels,
                        logits,
                        positive_weights=1.0,
                        negative_weights=1.0):
    """
    Computes weighted Hinge loss given predictions.
    :param labels: A torch.FloatTensor of shape [batch_size, num_classes] stored with ground truth in one hot.
    :param logits: A torch.FloatTensor of shape [batch_size, num_classes] stored with logits.
    :param positive_weights: A Tensor that holds positive weights.
    :param negative_weights: A Tensor that holds negative weights.
    :return: A torch.FloatTensor of the same shape as predictions with weighted Hinge loss.
    """
    positives_term = labels * (1 - logits).clamp(min=0)
    negatives_term = (1 - labels) * (1 + logits).clamp(min=0)
    # print(positives_term[:5], negatives_term[:5])
    return positive_weights * positives_term + negative_weights * negatives_term, positive_weights * positives_term


class LagrangeMultiplier(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return input.clamp(min=0)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg()

class ProbabilityLagrangeMultiplier(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return input.clamp(min=0, max=1)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg()


def lagrange_multiplier(x):
    return LagrangeMultiplier.apply(x)

def probability_lagrange_multiplier(x):
    return ProbabilityLagrangeMultiplier.apply(x)
    
  
def calibration_loss_bump_func(labels, probabilities, calibration_anchors, thetas):
    # In this method, we apply the bump function to smooth indicator function.

    # epsilon is a value tiny enough, such that it will only have arithmetic influence but not have influence for meaning.
    epsilon_val = 10 ** (-5)
    if torch.cuda.is_available():
        epsilon = torch.cuda.FloatTensor([epsilon_val])
    else:
        epsilon = torch.tensor([epsilon_val])
    # change all the probabilities to be the confidence of data to have a positive label
    if torch.cuda.is_available():
        probabilities = probabilities - torch.cuda.FloatTensor([0, 1.0])
        probabilities = probabilities * torch.cuda.FloatTensor([1.0, -1.0])
    else:
        probabilities = probabilities - torch.tensor([0, 1.0])
        probabilities = probabilities * torch.tensor([1.0, -1.0])

    lower_bound_calibration_anchors = calibration_anchors[:-1]
    upper_bound_calibration_anchors =calibration_anchors[1:]
    
    # add a tiny noise 'epsilon' to make the probablity value jump away from the boundary of the calibration slot.
    # I.e. to make the exact value of 0 to be a tiny positive value, to differentiate with the negative values after projection
    lower_bound_calibration_anchors_check = (probabilities.unsqueeze(-1) - lower_bound_calibration_anchors + epsilon).clamp(min = 0)
    upper_bound_calibration_anchors_check = (upper_bound_calibration_anchors - probabilities.unsqueeze(-1) - epsilon).clamp(min = 0)
    
    calibration_slot_satisfactory = lower_bound_calibration_anchors_check * upper_bound_calibration_anchors_check
    # divide by itself to make the value of exact 1 if the slot is satisfied. We add a tiny noise 'epsilon' as well, to get rid of the case of 0 / 0.
    # such tiny 'epsilon' has only arithmetic purpose without influencing semantic meaning.
    calibration_slot_satisfactory = calibration_slot_satisfactory / (calibration_slot_satisfactory + epsilon)
    # print("calibration_slot_satisfactory: ", calibration_slot_satisfactory)

    # calculation for bump func
    calibration_anchor_interval_radius = (upper_bound_calibration_anchors - lower_bound_calibration_anchors) / 2
    calibration_anchor_interval_center = (upper_bound_calibration_anchors + lower_bound_calibration_anchors) / 2
    deviation_from_center = probabilities.unsqueeze(-1) - calibration_anchor_interval_center
    # The reason for clamp here is to avoid the rescaled_deviation_from_center to be too close to the value of +1 or -1,
    # which may lead the part of 1 / (1 - rescaled_deviation_from_center ** 2) in the derivative of bump func
    # to explode to infty.
    # Thus, we use the value of epsilon to make it detach from the value of 1 or -1. The value of epsilon here
    # can neither be too tiny nor too large. 
    rescaled_deviation_from_center = (deviation_from_center / calibration_anchor_interval_radius).clamp(min = -1 + epsilon_val, max = 1 - epsilon_val)
    bump_func = torch.exp(-1 / (1 - rescaled_deviation_from_center ** 2)) * calibration_slot_satisfactory
    #bump_func = torch.exp(-1 / (1 - rescaled_deviation_from_center ** 2))
    #torch.set_printoptions(profile="full")
    #print("bump func val: ", bump_func)

    if torch.cuda.is_available():
        # following the paper, it lets theta minus 1 for positive sample point and lets theta minus 0 for negative sample point
        thetas_squared_loss = (thetas - torch.cuda.FloatTensor([1.0, 0]).unsqueeze(-1)) ** 2
    else:
        thetas_squared_loss = (thetas - torch.tensor([1.0, 0]).unsqueeze(-1)) ** 2
    
    calibration_loss = bump_func * thetas_squared_loss
    calibration_loss = (labels * calibration_loss.sum(-1)).sum(-1).mean()
    # print("calibration loss: ", calibration_loss)
    return calibration_loss
  
def calibration_loss_gaussian_conv(labels, probabilities, calibration_anchors, thetas):
    # In this method, instead of using indicator function, we apply guassian function with 
    # convolution operation over the indicator function. 

    # change all the probabilities to be the confidence of data to have a positive label
    if torch.cuda.is_available():
        probabilities = probabilities - torch.cuda.FloatTensor([0, 1.0])
        probabilities = probabilities * torch.cuda.FloatTensor([1.0, -1.0])
    else:
        probabilities = probabilities - torch.tensor([0, 1.0])
        probabilities = probabilities * torch.tensor([1.0, -1.0])

    lower_bound_calibration_anchors = calibration_anchors[:-1]
    upper_bound_calibration_anchors =calibration_anchors[1:]

    lower_bound_calibration_anchors_difference = probabilities.unsqueeze(-1) - lower_bound_calibration_anchors
    upper_bound_calibration_anchors_difference = probabilities.unsqueeze(-1) - upper_bound_calibration_anchors 

    convolution = 1.0/(2.0 * np.sqrt(2)) * (torch.erf(lower_bound_calibration_anchors_difference) - torch.erf(upper_bound_calibration_anchors_difference))
    
    if torch.cuda.is_available():
        # following the paper, it lets theta minus 1 for positive sample point and lets theta minus 0 for negative sample point
        thetas_squared_loss = (thetas - torch.cuda.FloatTensor([1.0, 0]).unsqueeze(-1)) ** 2
    else:
        thetas_squared_loss = (thetas - torch.tensor([1.0, 0]).unsqueeze(-1)) ** 2
   
    calibration_loss = convolution * thetas_squared_loss
    calibration_loss = (labels * calibration_loss.sum(-1)).sum(-1).mean()
    # print("calibration loss: ", calibration_loss)
    return calibration_loss

def masked_batch_alpha(data_indexes, alpha):
    batch_size = data_indexes.size()[0]
    train_set_size = alpha.size()[0]
    if torch.cuda.is_available():
        sparse_entry_indexes = torch.stack([torch.arange(0, batch_size).cuda(), data_indexes.cuda()])
        sparse_entry_val = torch.ones([batch_size]).cuda()
    else:
        sparse_entry_indexes = torch.stack([torch.arange(0, batch_size), data_indexes])
        sparse_entry_val = torch.ones([batch_size])

    # mask is a sparse tensor. 
    # Each row of mask is a one-hot bit encode of the batch's data's corresponding index from the whole training set.
    # e.g. for the first data in the batch, row 0 will represent for this data. 
    # Assume that the first data's index from the training set is 8,
    # then the one-hot bit encode of this data is entry of 1 at col 7, 
    # and all other entry of 0 in this row 0.
    mask = torch.sparse.FloatTensor(sparse_entry_indexes, sparse_entry_val, torch.Size([batch_size, train_set_size]))
    return torch.sparse.mm(mask, alpha)
    
