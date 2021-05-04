import loss_utils
import torch.nn as nn
import torch

class AUCROCLoss(nn.Module):

    def __init__(self, config, device):
        super(AUCROCLoss, self).__init__()
        self.config = config
        self.num_classes = self.config['num_classes']
        self.num_anchors = self.config['num_anchors']
        self.rate_range = self.config['rate_range']
        self.device = device
        # self.class_priors = self.config['class_priors']
        self.rate_values, self.delta = loss_utils.range_to_anchors_and_delta(self.rate_range, self.num_anchors, device)

        self.use_fpr_at_tpr = self.config['mode'] == "F@T"
        if self.use_fpr_at_tpr:
            self.tpr_values = self.rate_values
        else:
            self.fpr_values = self.rate_values

    def forward(self, labels, logits, probabilities, dual_var, bias, device, weights=None):
        labels, weights = loss_utils.prepare_labels_weights(logits, labels, device, weights=weights)
        class_priors = loss_utils.build_class_priors(labels, device)

        lambdas = loss_utils.lagrange_multiplier(dual_var)
            
        if self.use_fpr_at_tpr:
            hinge_loss, check_loss = loss_utils.weighted_hinge_loss(
                labels.unsqueeze(-1),
                logits.unsqueeze(-1) - bias,
                positive_weights=lambdas,
                negative_weights=1.0)

            lambda_term = lambdas * (1.0 - self.tpr_values) * class_priors.unsqueeze(-1) 

        else:
            hinge_loss, check_loss = loss_utils.weighted_hinge_loss(
                labels.unsqueeze(-1),
                logits.unsqueeze(-1) - bias,
                positive_weights=1.0,
                negative_weights=lambdas)

            lambda_term = (1 - class_priors).unsqueeze(-1) * lambdas * self.fpr_values

        # print(weights.unsqueeze(-1).shape, hinge_loss.shape, lambda_term.shape)
        # exit()
        
        per_anchor_loss = weights.unsqueeze(-1) * hinge_loss - lambda_term
        loss = per_anchor_loss.sum(2) * self.delta
        per_anchor_check_loss = weights.unsqueeze(-1) * check_loss - lambda_term
        # print(per_anchor_check_loss.sum(2) <= 0)
        
        #print("logits shape: ", logits.shape)
        #print("loss shape: ", loss.shape)
        #print("result shape: ", loss.mean().shape)
        
        # Normalize over precision range
        loss /= self.rate_range[1] - self.rate_range[0]

        return loss.mean()
