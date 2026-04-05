import os
import math
import torch
import torch.nn as nn
import networkx as nx
import dgl
from functools import partial
from collections import Counter
import dgl.function as fn
import random
import numpy as np
import sympy
import torch.nn.functional as F
import scipy



def extract_key_nodes_for_class(model, distances, target_class, k):
    proto_indices = torch.where(model.prototype_class_identity[:, target_class])[0]
    dists_to_class_protos = distances[:, proto_indices].min(dim=1)[0]
    _, topk_indices = torch.topk(dists_to_class_protos, k, largest=False)
    return topk_indices


def calculate_theta2(d):
    thetas = []
    x = sympy.symbols('x')
    for i in range(d+1):
        f = sympy.poly((x/2) ** i * (1 - x/2) ** (d-i) / (scipy.special.beta(i+1, d+1-i)))
        coeff = f.all_coeffs()
        inv_coeff = []
        for i in range(d+1):
            inv_coeff.append(float(coeff[d-i]))
        thetas.append(inv_coeff)
    return thetas



class PolyConv(nn.Module):
    def __init__(self,
                 in_feats,
                 out_feats,
                 theta,
                 activation=F.leaky_relu,
                 lin=False,
                 bias=False):
        super(PolyConv, self).__init__()
        self._theta = theta
        self._k = len(self._theta)
        self._in_feats = in_feats
        self._out_feats = out_feats
        self.activation = activation
        self.linear = nn.Linear(in_feats, out_feats, bias)
        self.lin = lin
        # self.reset_parameters()
        # self.linear2 = nn.Linear(out_feats, out_feats, bias)

    def reset_parameters(self):
        if self.linear.weight is not None:
            init.xavier_uniform_(self.linear.weight)
        if self.linear.bias is not None:
            init.zeros_(self.linear.bias)

    def forward(self, graph, feat):
        def unnLaplacian(feat, D_invsqrt, graph):
            """ Operation Feat * D^-1/2 A D^-1/2 """
            graph.ndata['h'] = feat * D_invsqrt
            graph.update_all(fn.copy_u('h', 'm'), fn.sum('m', 'h'))
            return feat - graph.ndata.pop('h') * D_invsqrt

        with graph.local_scope():
            D_invsqrt = torch.pow(graph.in_degrees().float().clamp(
                min=1), -0.5).unsqueeze(-1).to(feat.device)
            h = self._theta[0]*feat
            for k in range(1, self._k):
                feat = unnLaplacian(feat, D_invsqrt, graph)
                h += self._theta[k]*feat
        if self.lin:
            h = self.linear(h)
            h = self.activation(h)
        return h
