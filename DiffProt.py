import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl.function as fn
import math
import dgl
import sympy
import scipy
import numpy as np
import json
from torch import nn
from torch_geometric.nn.glob import global_mean_pool, global_add_pool, global_max_pool

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
import dgl.nn.pytorch as dglnn
from dgl.nn.pytorch.conv import SAGEConv, GraphConv
from utils import *



class SiLU(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)

class PositionalEmbedding(torch.nn.Module):
    def __init__(self, num_channels, max_positions=10000, endpoint=False):
        super().__init__()
        self.num_channels = num_channels
        self.max_positions = max_positions
        self.endpoint = endpoint

    def forward(self, x):
        freqs = torch.arange(start=0, end=self.num_channels // 2,
                             dtype=torch.float32, device=x.device)
        freqs = freqs / (self.num_channels // 2 - (1 if self.endpoint else 0))
        freqs = (1 / self.max_positions) ** freqs
        x = x.ger(freqs.to(x.dtype))
        x = torch.cat([x.cos(), x.sin()], dim=1)
        return x

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = torch.log(torch.tensor(10000.0, device=device)) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        x_2d = x.view(-1, 1) 
        emb = x_2d * emb.unsqueeze(0)
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb

class EDMDenoiseNet(nn.Module):
    def __init__(self, d_in, num_classes, time_dim=128, cond_dim=128):
        super().__init__()
        self.dim_t = time_dim
        self.d_in = d_in

        self.map_noise = PositionalEmbedding(num_channels=time_dim)
        self.time_embed = nn.Sequential(
            nn.Linear(time_dim, time_dim), 
            nn.SiLU(),
            nn.Linear(time_dim, time_dim)
        )
        
        self.class_emb = nn.Embedding(num_classes, cond_dim)
        
        self.inconsistency_mlp = nn.Sequential(
            SinusoidalPosEmb(cond_dim),
            nn.Linear(cond_dim, cond_dim * 4), SiLU(),
            nn.Linear(cond_dim * 4, cond_dim),
        )

        self.proj = nn.Linear(d_in, time_dim)

        self.main_net = nn.Sequential(
            nn.Linear(time_dim + cond_dim + cond_dim, d_in * 2), 
            nn.SiLU(),
            nn.Linear(d_in * 2, d_in * 2), 
            nn.SiLU(),
            nn.Linear(d_in * 2, d_in)
        )

    def forward(self, x, c_noise, labels, inconsistency_score):
        t_emb = self.map_noise(c_noise)
        t_emb = self.time_embed(t_emb)
        
        f_emb = self.inconsistency_mlp(inconsistency_score)
        x_proj = self.proj(x) + t_emb
        if labels!=None:
            c_emb = self.class_emb(labels)
        else:
            c_emb = torch.zeros(x.size(0),self.class_emb.embedding_dim,device=x.device,dtype=x.dtype)

        h = torch.cat([x_proj, c_emb, f_emb], dim=1)
        return self.main_net(h)


class Precond(nn.Module):
    def __init__(self,
                 denoise_fn,
                 sigma_data=0.5,
                 ):
        super().__init__()
        self.sigma_data = sigma_data
        self.denoise_fn_F = denoise_fn

    def forward(self, x, sigma, labels=None, inconsistency_score=None):
        x = x.to(torch.float32)
        sigma = sigma.to(torch.float32).reshape(-1, 1)
        dtype = torch.float32
        c_skip = self.sigma_data ** 2 / (sigma ** 2 + self.sigma_data ** 2)
        c_out = sigma * self.sigma_data / (
            sigma ** 2 + self.sigma_data ** 2).sqrt()
        c_in = 1 / (self.sigma_data ** 2 + sigma ** 2).sqrt()
        c_noise = sigma.log() / 4 

        x_in = c_in * x
        
        F_x = self.denoise_fn_F(x_in.to(dtype), c_noise.flatten(), labels, inconsistency_score)

        assert F_x.dtype == dtype
        D_x = c_skip * x + c_out * F_x.to(torch.float32)
        return D_x


SIGMA_MIN = 0.002
SIGMA_MAX = 80
rho = 7
S_churn = 1
S_min = 0
S_max = float('inf')
S_noise = 1

def sample_step(net, num_steps, i, t_cur, t_next, x_next, cond_kwargs):
    x_cur = x_next
    gamma = min(S_churn / num_steps, math.sqrt(2) - 1) if S_min <= t_cur <= S_max else 0
    t_hat = t_cur + gamma * t_cur
    x_hat = x_cur + (t_hat ** 2 - t_cur ** 2).sqrt() * S_noise * torch.randn_like(x_cur)
    
    denoised = net(x_hat, t_hat, **cond_kwargs).to(torch.float32)
    d_cur = (x_hat - denoised) / t_hat
    x_next = x_hat + (t_next - t_hat) * d_cur

    if i < num_steps - 1:
        denoised = net(x_next, t_next, **cond_kwargs).to(torch.float32)
        d_prime = (x_next - denoised) / t_next
        x_next = x_hat + (t_next - t_hat) * (0.5 * d_cur + 0.5 * d_prime)

    return x_next


def sample_dm(net, noise, num_steps, cond_kwargs={}):
    step_indices = torch.arange(num_steps, dtype=torch.float32, device=noise.device)

    sigma_min = SIGMA_MIN
    sigma_max = SIGMA_MAX

    t_steps = (sigma_max ** (1 / rho) + step_indices / (num_steps - 1) * (
        sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho
    t_steps = torch.cat([t_steps, torch.zeros_like(t_steps[:1])]) # t_N, ..., t_0

    z = noise.to(torch.float32) * t_steps[0]

    with torch.no_grad():
        for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])):
            z = sample_step(net, num_steps, i, t_cur, t_next, z, cond_kwargs)
    return z


def load_config(path):
    """Load JSON config file and return as dict."""
    with open(path, 'r') as f:
        return json.load(f)


class DiffProt(nn.Module):
    def __init__(self, in_feats, args):
        super(DiffProt, self).__init__()

        h_feats = args.h_feats
        device = args.device
        encoder = args.encoder

        self.P_mean = args.P_mean
        self.P_std = args.P_std
        self.sigma_data = args.sigma_data

        self.latent_dim_out = h_feats
        self.output_dim = args.num_classes
        self.num_prototypes_per_class = args.num_prototypes_per_class
        self.device = device
        self.epsilon = 1e-4
        
        self.num_prototypes = self.num_prototypes_per_class * self.output_dim

        self.encoder = encoder.lower()
        self.convs = nn.ModuleList()

        if self.encoder in ('gcn'):
            for _ in range(args.order):
                self.convs.append(GraphConv(h_feats, h_feats))
        elif self.encoder in ('graphsage'):
            for _ in range(args.order):
                self.convs.append(SAGEConv(h_feats, h_feats, 'mean'))
        elif self.encoder in ('bwgnn'):
            self.thetas = calculate_theta2(d=args.order)
            for i in range(len(self.thetas)):
                self.convs.append(PolyConv(h_feats, h_feats, self.thetas[i], lin=False))
        else:
            print(f"Unknown encoder type '{encoder}'")
            exit(1)

        self.linear = nn.Linear(in_feats, h_feats)
        self.linear2 = nn.Linear(h_feats, h_feats)
        self.linear3 = nn.Linear(h_feats * len(self.convs), h_feats)
        self.linear4 = nn.Linear(h_feats + self.num_prototypes, self.output_dim)
        self.act = nn.ReLU()

        self.denoise_net_aug = EDMDenoiseNet(
            d_in=self.num_prototypes + self.latent_dim_out,
            num_classes=self.output_dim,
        ).to(self.device)

        self.denoise_fn_D = Precond(
            denoise_fn=self.denoise_net_aug,
            sigma_data=self.sigma_data
        ).to(self.device)

        self.denoise_net_prot = EDMDenoiseNet(
            d_in=self.latent_dim_out,
            num_classes=self.output_dim,
        ).to(self.device)

        self.denoise_fn_D_prot = Precond(
            denoise_fn=self.denoise_net_prot,
            sigma_data=self.sigma_data
        ).to(self.device)

        # [K_total, D]
        self.prototypes = nn.Parameter(
            torch.randn(self.num_prototypes, self.latent_dim_out),
            requires_grad=True
        )

        # class identity for prototypes (flat index)
        prototype_class_identity = torch.zeros(self.num_prototypes, self.output_dim, device=self.device)
        for j in range(self.num_prototypes):
            prototype_class_identity[j, j // self.num_prototypes_per_class] = 1

        self.prototype_labels = prototype_class_identity.argmax(dim=1)
        self.register_buffer('prototype_class_identity', prototype_class_identity)

        self.set_last_layer_incorrect_connection(incorrect_strength=-0.5)


    def set_last_layer_incorrect_connection(self, incorrect_strength):
        positive_one_weights_locations = torch.t(self.prototype_class_identity)
        negative_one_weights_locations = 1 - positive_one_weights_locations

        correct_class_connection = 1
        incorrect_class_connection = incorrect_strength
        self.linear4.weight[:,: self.num_prototypes].data.copy_(
            correct_class_connection * positive_one_weights_locations
            + incorrect_class_connection * negative_one_weights_locations)

    def compute_local_inconsistency(self, graph, x):
        with graph.local_scope():
            graph.ndata['h'] = x
            graph.update_all(fn.copy_u('h', 'm'), fn.mean('m', 'h_neigh'))
            diff = x - graph.ndata['h_neigh']
            score = torch.norm(diff, p=2, dim=1)
            return score
        
    def prototype_distances(self, x, prototypes):
        xp = torch.mm(x, prototypes.t())
        x_sq = torch.sum(x ** 2, dim=1, keepdim=True)
        p_sq = torch.sum(prototypes ** 2, dim=1, keepdim=True).t()
        distance = -2 * xp + x_sq + p_sq
        similarity = torch.log((distance + 1) / (distance + self.epsilon))
        return similarity, distance
    
    def forward(self, in_feat, graph):
        # encoder
        h = self.linear(in_feat)
        h = self.act(h)
        h = self.linear2(h)
        h = self.act(h)
        h_final = torch.zeros([len(in_feat), 0]).to(h.device)
        for conv in self.convs:
            h0 = conv(graph, h)
            h_final = torch.cat([h_final, h0], -1)
        h = self.linear3(h_final)
        h = self.act(h)

        # prototype
        prototype_activations, min_distances = self.prototype_distances(h, self.prototypes)
        
        # concat
        final_input = torch.cat((prototype_activations, h), dim=1)

        h_logits = self.linear4(final_input)
        
        return h_logits, final_input, min_distances

    def train_diffusion_step(self, graph, x, labels=None):
        rnd_normal = torch.randn(x.shape[0], device=self.device)
        sigma = (rnd_normal * self.P_std + self.P_mean).exp()
        weight = (sigma ** 2 + self.sigma_data ** 2) / \
                 (sigma * self.sigma_data) ** 2

        target = x
        n = torch.randn_like(target) * sigma.unsqueeze(1)
        yn = target + n

        i_t = self.compute_local_inconsistency(graph, yn)
        
        D_yn = self.denoise_fn_D(yn, sigma, labels, i_t)

        i_pred = self.compute_local_inconsistency(graph, D_yn)
        
        recon_loss = weight.unsqueeze(1) * ((D_yn - target) ** 2)
        consistency_loss = F.mse_loss(i_pred, i_t)

        return recon_loss.mean(), consistency_loss

    def prototype_loss(self, current_node_emb):
        current_node_emb = current_node_emb[:, self.num_prototypes:]
        z_prototypes = self.prototypes # (num_prototypes, h_feats)
        prototype_labels = self.prototype_labels.to(self.device)
        
        rnd_normal = torch.randn(self.num_prototypes, device=self.device)
        sigma = (rnd_normal * self.P_std + self.P_mean).exp()
        weight = (sigma ** 2 + self.sigma_data ** 2) / (sigma * self.sigma_data) ** 2

        target = z_prototypes
        n = torch.randn_like(target) * sigma.unsqueeze(1)
        D_prototypes = target + n

        with torch.no_grad():
            dist_matrix = torch.cdist(D_prototypes, current_node_emb, p=2)
            k_neigh = 10
            topk_dists, _ = torch.topk(dist_matrix, k=k_neigh, dim=1, largest=False)
            proto_self_inconsistency = topk_dists.mean(dim=1) 

        D_pn = self.denoise_fn_D_prot(
            D_prototypes, 
            sigma, 
            prototype_labels, 
            proto_self_inconsistency 
        )
        
        loss = weight.unsqueeze(1) * ((D_pn - target) ** 2)
        
        dists_post = torch.cdist(D_pn, current_node_emb, p=2)
        topk_post, _ = torch.topk(dists_post, k=k_neigh, dim=1, largest=False)
        struct_loss = topk_post.mean()

        return loss.mean(), struct_loss


class DiffProt_here(nn.Module):
    def __init__(self, in_feats, args):
        super(DiffProt_here, self).__init__()

        h_feats = args.h_feats
        device = args.device
        encoder = getattr(args, 'encoder_type', getattr(args, 'encoder', 'BWGNN'))

        self.P_mean = args.P_mean
        self.P_std = args.P_std
        self.sigma_data = args.sigma_data

        self.latent_dim_out = h_feats
        self.output_dim = args.num_classes
        self.num_prototypes_per_class = args.num_prototypes_per_class
        self.device = device
        self.epsilon = 1e-4

        self.num_rels = args.num_rels 

        self.num_prototypes_per_rel = self.output_dim * self.num_prototypes_per_class
        self.num_prototypes = self.num_rels * self.num_prototypes_per_rel
        self.rel_weights = nn.Parameter(torch.ones(self.num_rels) / self.num_rels)

        self.encoder = encoder.lower()
        self.convs = nn.ModuleList()

        if self.encoder in ('gcn'):
            for _ in range(args.order):
                self.convs.append(GraphConv(h_feats, h_feats))
        elif self.encoder in ('graphsage'):
            for _ in range(args.order):
                self.convs.append(SAGEConv(h_feats, h_feats, 'mean'))
        elif self.encoder in ('bwgnn'):
            self.thetas = calculate_theta2(d=args.order)
            for i in range(len(self.thetas)):
                self.convs.append(PolyConv(h_feats, h_feats, self.thetas[i], lin=False))
        else:
            print(f"Unknown encoder type '{encoder}'")
            exit(1)

        self.linear = nn.Linear(in_feats, h_feats)
        self.linear2 = nn.Linear(h_feats, h_feats)
        self.linear3 = nn.Linear(h_feats * len(self.convs), h_feats)
        self.linear4 = nn.Linear(h_feats + self.num_prototypes, self.output_dim)
        self.act = nn.ReLU()

        self.denoise_net_aug = EDMDenoiseNet(
            d_in=self.num_prototypes + self.latent_dim_out,
            num_classes=self.output_dim,
        ).to(self.device)

        self.denoise_fn_D = Precond(
            denoise_fn=self.denoise_net_aug,
            sigma_data=self.sigma_data
        ).to(self.device)

        self.denoise_net_prot = EDMDenoiseNet(
            d_in=self.latent_dim_out,
            num_classes=self.output_dim,
        ).to(self.device)

        self.denoise_fn_D_prot = Precond(
            denoise_fn=self.denoise_net_prot,
            sigma_data=self.sigma_data
        ).to(self.device)

        # [K_total, D]
        self.prototypes = nn.Parameter(
            torch.randn(self.num_prototypes, self.latent_dim_out),
            requires_grad=True
        )

        # class identity for prototypes (flat index)
        prototype_class_identity = torch.zeros(self.num_prototypes, self.output_dim, device=self.device)
        # for hetero, assign per-relation blocks
        for e in range(self.num_rels):
            for j in range(self.num_prototypes_per_rel):
                global_idx = e * self.num_prototypes_per_rel + j
                class_label = j // self.num_prototypes_per_class
                prototype_class_identity[global_idx, class_label] = 1

        self.prototype_labels = prototype_class_identity.argmax(dim=1)
        self.register_buffer('prototype_class_identity', prototype_class_identity)

        self.set_last_layer_incorrect_connection(incorrect_strength=-0.5)


    def set_last_layer_incorrect_connection(self, incorrect_strength):
        positive_one_weights_locations = torch.t(self.prototype_class_identity)
        negative_one_weights_locations = 1 - positive_one_weights_locations

        correct_class_connection = 1
        incorrect_class_connection = incorrect_strength
        self.linear4.weight[:,: self.num_prototypes].data.copy_(
            correct_class_connection * positive_one_weights_locations
            + incorrect_class_connection * negative_one_weights_locations)

    def compute_local_inconsistency(self, graph, x):
        with graph.local_scope():
            ntype = graph.ntypes[0]  # 唯一的节点类型
            graph.nodes[ntype].data['h'] = x
            # 为每种边类型构建消息传递函数
            funcs = {}
            for etype in graph.canonical_etypes:
                funcs[etype] = (fn.copy_u('h', 'm'), fn.mean('m', 'h_neigh'))
            # cross_reducer 决定不同边类型的聚合结果如何合并
            graph.multi_update_all(funcs, cross_reducer='mean')
            h_neigh = graph.nodes[ntype].data['h_neigh']
            diff = x - h_neigh
            score = torch.norm(diff, p=2, dim=1)
            return score
        
    def prototype_distances(self, x, prototypes):
        xp = torch.mm(x, prototypes.t())
        x_sq = torch.sum(x ** 2, dim=1, keepdim=True)
        p_sq = torch.sum(prototypes ** 2, dim=1, keepdim=True).t()
        distance = -2 * xp + x_sq + p_sq
        similarity = torch.log((distance + 1) / (distance + self.epsilon))
        return similarity, distance
    
    def forward(self, in_feat, graph):
        # encoder
        h = self.linear(in_feat)
        h = self.act(h)
        h = self.linear2(h)
        h = self.act(h)

        h_gnn_all = []
        prototype_activations_all = []
        min_distances_all = []
        prototypes_view = self.prototypes.view(self.num_rels, self.num_prototypes_per_rel, self.latent_dim_out)

        for i, relation in enumerate(graph.canonical_etypes):
            h_final = torch.zeros([len(in_feat), 0]).to(h.device)
            for conv in self.convs:
                h0 = conv(graph[relation], h)
                h_final = torch.cat([h_final, h0], -1)

            h_rel_gnn = self.linear3(h_final)
            h_gnn_all.append(h_rel_gnn)

            current_prototypes = prototypes_view[i]
            proto_act_rel, min_dist_rel = self.prototype_distances(h_rel_gnn, current_prototypes)
            prototype_activations_all.append(proto_act_rel)
            min_distances_all.append(min_dist_rel)

        h_stacked = torch.stack(h_gnn_all).sum(0)
        weights = torch.softmax(self.rel_weights, dim=0).view(-1, 1, 1)
        h_aggregated = (h_stacked * weights).sum(dim=0)
        h = self.act(h_aggregated)

        # prototype
        prototype_activations = torch.cat(prototype_activations_all, dim=1)
        min_distances = torch.cat(min_distances_all, dim=1)
        
        # concat
        final_input = torch.cat((prototype_activations, h), dim=1)

        h_logits = self.linear4(final_input)
        
        return h_logits, final_input, min_distances

    def train_diffusion_step(self, graph, x, labels=None):
        rnd_normal = torch.randn(x.shape[0], device=self.device)
        sigma = (rnd_normal * self.P_std + self.P_mean).exp()
        weight = (sigma ** 2 + self.sigma_data ** 2) / \
                 (sigma * self.sigma_data) ** 2

        target = x
        n = torch.randn_like(target) * sigma.unsqueeze(1)
        yn = target + n

        i_t = self.compute_local_inconsistency(graph, yn)
        
        D_yn = self.denoise_fn_D(yn, sigma, labels, i_t)

        i_pred = self.compute_local_inconsistency(graph, D_yn)
        
        recon_loss = weight.unsqueeze(1) * ((D_yn - target) ** 2)
        consistency_loss = F.mse_loss(i_pred, i_t)

        return recon_loss.mean(), consistency_loss

    def prototype_loss(self, current_node_emb):
        current_node_emb = current_node_emb[:, self.num_prototypes:]
        z_prototypes = self.prototypes # (num_prototypes, h_feats)
        prototype_labels = self.prototype_labels.to(self.device)
        
        rnd_normal = torch.randn(self.num_prototypes, device=self.device)
        sigma = (rnd_normal * self.P_std + self.P_mean).exp()
        weight = (sigma ** 2 + self.sigma_data ** 2) / (sigma * self.sigma_data) ** 2

        target = z_prototypes
        n = torch.randn_like(target) * sigma.unsqueeze(1)
        D_prototypes = target + n

        with torch.no_grad():
            dist_matrix = torch.cdist(D_prototypes, current_node_emb, p=2)
            k_neigh = 10
            topk_dists, _ = torch.topk(dist_matrix, k=k_neigh, dim=1, largest=False)
            proto_self_inconsistency = topk_dists.mean(dim=1) 

        D_pn = self.denoise_fn_D_prot(
            D_prototypes, 
            sigma, 
            prototype_labels, 
            proto_self_inconsistency 
        )
        
        loss = weight.unsqueeze(1) * ((D_pn - target) ** 2)
        
        dists_post = torch.cdist(D_pn, current_node_emb, p=2)
        topk_post, _ = torch.topk(dists_post, k=k_neigh, dim=1, largest=False)
        struct_loss = topk_post.mean()

        return loss.mean(), struct_loss