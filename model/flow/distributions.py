import torch
from torch import distributions
import torch.nn.functional as F
import numpy as np
import math



class FastGMM(torch.distributions.Distribution):
# This implementation is to achieve fast Gaussian mixture with vectorized calcualtion
# Note that the vectorized method only applies for gaussian with diagonal covariance matrix
 
    def __init__(self, means, unlabeled=300):
        self.n_components, self.dim = means.shape
        self.means = means
        
        # Diagonal covariance matrix with ones
        self.covs = torch.ones(self.n_components).unsqueeze(1).cuda()

        # Weights for each component
        self.weights = torch.ones(self.n_components).cuda()
        self.unlabeled = unlabeled
        self.log_pi_const = torch.tensor(self.dim*math.log(2*math.pi)).cuda()


    def log_prob_base(self, x):
        # Calculate log likelihood of x given each distribution
        # Prepare
        covs_det = torch.prod(self.covs, dim=1)
        covs_inv = 1. / self.covs
        x_centered = x.unsqueeze(1).repeat(1, self.n_components, 1) - self.means.unsqueeze(0) # n_x, ndist, dim

        mahala_dist = torch.sum(x_centered**2 * covs_inv.unsqueeze(0), dim=2) # n_x, ndist
        log_cov_det = covs_det.log().unsqueeze(0).repeat(x.size(0), 1) # n_x, n_dist

        # Log prob
        all_log_probs = -0.5*(self.log_pi_const + mahala_dist + log_cov_det) # n_x, n_dist
        
        return all_log_probs
        

    def log_prob(self, x, y=None):
        all_log_probs = self.log_prob_base(x)
        normalized_weights = F.softmax(self.weights, dim=0)
        mixture_log_probs = torch.logsumexp(all_log_probs + torch.log(normalized_weights), dim=1)

        if y is not None:
            log_probs = torch.zeros_like(mixture_log_probs)
            mask = (y == self.unlabeled) # For unlabled data, this is equivalent to unsupervised learning
            log_probs[mask] += mixture_log_probs[mask] # It shows that this part is really important for the model to be able to predict
            for i in range(self.n_components):
                #Pavel: add class weights here? 
                mask = (y == i)
                # log_probs[mask] += all_log_probs[:, i][mask] + torch.log(normalized_weights[i])
                log_probs[mask] += all_log_probs[:, i][mask]
            return log_probs
        else:
            return mixture_log_probs

    @property
    def gaussians(self):
        gaussians = [distributions.MultivariateNormal(mean, cov*torch.eye(self.dim).cuda())
                          for mean, cov in zip(self.means, self.covs)]
        return gaussians

        
    def class_logits(self, x):
        log_probs = self.log_prob_base(x)
        log_probs_weighted = log_probs + torch.log(F.softmax(self.weights, dim=0))
        return log_probs_weighted

    def classify(self, x):
        log_probs = self.class_logits(x)
        return torch.argmax(log_probs, dim=1)

    def class_probs(self, x):
        log_probs = self.class_logits(x)
        return F.softmax(log_probs, dim=1)