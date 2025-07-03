#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
@author: Vencent_Wang
@contact: Vencent_Wang@outlook.com
@file: multi_modal.py
@time: 2023/8/13 20:05
@desc:
'''
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import global_mean_pool, GlobalAttention
from models_lib.gnn_model import MPNEncoder
from models_lib.gem_model import GeoGNNModel
from models_lib.seq_model import TrfmSeq2seq

loss_type = {'class': nn.BCEWithLogitsLoss(reduction="none"), 'reg': nn.MSELoss(reduction="none")}

class Global_Attention(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.at = GlobalAttention(gate_nn=torch.nn.Linear(hidden_size, 1))

    def forward(self, x, batch):
        return self.at(x, batch)

# class DoubleWeightFusion(nn.Module):
class DoubleWeightFusion(nn.Module):
    """
    多模态门控+多头注意力融合。
    输入: shared_list, 每个元素 shape [batch, shared_dim]
    输出: 融合后的特征 [batch, out_dim]
    """
    def __init__(self, num_modalities, shared_dim=192, out_dim=192, num_heads=4, dropout=0.1, bias=True, device=None, dtype=None):
        super().__init__()
        self.num_modalities = num_modalities
        self.shared_dim = shared_dim
        self.out_dim = out_dim
        self.num_heads = num_heads

        self.attn = nn.MultiheadAttention(embed_dim=shared_dim, num_heads=num_heads, dropout=dropout, batch_first=True)
        self.proj = nn.Linear(shared_dim, out_dim, bias=bias)
        self.dropout = nn.Dropout(dropout)
        # 门控参数，每个模态一个gate
        self.gate = nn.Parameter(torch.zeros(num_modalities))

    def forward(self, shared_list):
        device = shared_list[0].device
        self.attn.to(device)
        self.proj.to(device)
        self.dropout.to(device)
        # self.gate = self.gate.to(device)  # 不要这样！

        x = torch.stack([s.to(device) for s in shared_list], dim=1)  # [batch, num_modalities, shared_dim]
        attn_out, _ = self.attn(x, x, x)  # [batch, num_modalities, shared_dim]

        # 门控权重，softmax归一化
        gate_weights = torch.softmax(self.gate, dim=0).to(device)  # [num_modalities] -> [num_modalities] on correct device
        gate_weights = gate_weights.view(1, -1, 1)      # [1, num_modalities, 1]

        fused = (attn_out * gate_weights).sum(dim=1)    # [batch, shared_dim]
        fused = self.dropout(fused)
        fused = self.proj(fused)
        return fused

# class WeightedSumFusion(nn.Module):
# class DoubleWeightFusion(nn.Module):
#     """
#     普通加权和融合：每个模态一个可学习权重和偏置。
#     输入: shared_list, 每个元素 shape [batch, shared_dim]
#     输出: 融合后的特征 [batch, out_dim]
#     """
#     def __init__(self, num_modalities, shared_dim=192, out_dim=192, device=None, dtype=None):
#         super().__init__()
#         self.num_modalities = num_modalities
#         self.shared_dim = shared_dim
#         self.out_dim = out_dim
#         self.weights = nn.Parameter(torch.ones(num_modalities))
#         self.proj = nn.Linear(shared_dim, out_dim, bias=True)
#         if device is not None:
#             self.to(device)

#     def forward(self, shared_list):
#         x = torch.stack(shared_list, dim=1)  # [batch, num_modalities, shared_dim]
#         device = x.device
#         w = torch.softmax(self.weights, dim=0).to(device).view(1, -1, 1)  # [1, num_modalities, 1]
#         fused = (x * w).sum(dim=1)  # [batch, shared_dim]
#         # NaN检查
#         if torch.isnan(fused).any():
#             raise ValueError("NaN detected in fusion output!")
#         fused = self.proj(fused)
#         return fused

# class ConcatMLPFusion(nn.Module):
# class DoubleWeightFusion(nn.Module):
#     """
#     拼接+MLP融合：将所有模态特征拼接后用MLP降维。
#     """
#     def __init__(self, num_modalities, shared_dim=192, out_dim=192, num_heads=4, dropout=0.1, bias=True, device=None, dtype=None, hidden_dim=256, norm=True):
#         super().__init__()
#         mlp_layers = [
#             nn.Linear(num_modalities * shared_dim, hidden_dim, bias=bias),
#             nn.ReLU()
#         ]
#         if norm:
#             mlp_layers.append(nn.LayerNorm(hidden_dim))
#         mlp_layers.append(nn.Dropout(dropout))
#         mlp_layers.append(nn.Linear(hidden_dim, out_dim, bias=bias))
#         self.mlp = nn.Sequential(*mlp_layers)
#         if device is not None:
#             self.mlp.to(device)

#     def forward(self, shared_list):
#         device = self.mlp[0].weight.device
#         x = torch.cat([s.to(device) for s in shared_list], dim=1)
#         fused = self.mlp(x)
#         return fused

class VAEHead(nn.Module):
    """
    多模态特征解耦的VAE模块，输出共享/私有隐变量及其高斯参数，并支持重参数化采样。
    """
    def __init__(self, in_dim, shared_dim, private_dim, hidden_dim=512, dropout=0.1, out_act=None, norm=True):
        super().__init__()
        encoder_layers = [
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU()
        ]
        if norm:
            encoder_layers.append(nn.LayerNorm(hidden_dim))
        encoder_layers.append(nn.Dropout(dropout))
        self.encoder = nn.Sequential(*encoder_layers)

        self.fc_mu_shared = nn.Linear(hidden_dim, shared_dim)
        self.fc_logvar_shared = nn.Linear(hidden_dim, shared_dim)
        self.fc_mu_private = nn.Linear(hidden_dim, private_dim)
        self.fc_logvar_private = nn.Linear(hidden_dim, private_dim)

        decoder_layers = [
            nn.Linear(shared_dim + private_dim, hidden_dim),
            nn.ReLU()
        ]
        if norm:
            decoder_layers.append(nn.LayerNorm(hidden_dim))
        decoder_layers.append(nn.Dropout(dropout))
        decoder_layers.append(nn.Linear(hidden_dim, in_dim))
        self.decoder = nn.Sequential(*decoder_layers)
        self.out_act = out_act

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        else:
            return mu

    def forward(self, x):
        h = self.encoder(x)
        mu_shared = self.fc_mu_shared(h)
        # clamp logvar，防止数值溢出
        logvar_shared = torch.clamp(self.fc_logvar_shared(h), min=-10, max=10)
        mu_private = self.fc_mu_private(h)
        logvar_private = torch.clamp(self.fc_logvar_private(h), min=-10, max=10)
        z_shared = self.reparameterize(mu_shared, logvar_shared)
        z_private = self.reparameterize(mu_private, logvar_private)
        z = torch.cat([z_shared, z_private], dim=1)
        recon_x = self.decoder(z)
        # if self.out_act == 'sigmoid':
        #     recon_x = torch.sigmoid(recon_x)
        # NaN检查
        if torch.isnan(z_shared).any() or torch.isnan(z_private).any():
            raise ValueError("NaN detected in VAE latent variables!")
        return z_shared, z_private, recon_x, mu_shared, logvar_shared, mu_private, logvar_private

class Multi_modal(nn.Module):
    def __init__(self, args, compound_encoder_config, device):
        super().__init__()
        self.args = args
        self.device = device
        self.latent_dim = args.latent_dim
        self.batch_size = args.batch_size
        self.graph = args.graph
        self.sequence = args.sequence
        self.geometry = args.geometry
        self.shared_dim = args.latent_dim // 4 * 3
        self.private_dim = args.latent_dim // 4

        self.gnn = MPNEncoder(atom_fdim=args.gnn_atom_dim, bond_fdim=args.gnn_bond_dim,
                              hidden_size=args.gnn_hidden_dim, bias=args.bias, depth=args.gnn_num_layers,
                              dropout=args.dropout, activation=args.gnn_activation, device=device)
        self.transformer = TrfmSeq2seq(input_dim=args.seq_input_dim, hidden_size=args.seq_hidden_dim,
                                       num_head=args.seq_num_heads, n_layers=args.seq_num_layers, dropout=args.dropout,
                                       vocab_num=args.vocab_num, device=device, recons=args.recons).to(self.device)
        self.compound_encoder = GeoGNNModel(args, compound_encoder_config, device)

        self.gnn_ae = VAEHead(args.gnn_hidden_dim, self.shared_dim, self.private_dim, norm=bool(args.norm)).to(device)
        self.seq_ae = VAEHead(args.seq_hidden_dim, self.shared_dim, self.private_dim, norm=bool(args.norm)).to(device)
        self.geo_ae = VAEHead(args.geo_hidden_dim, self.shared_dim, self.private_dim, norm=bool(args.norm)).to(device)

        if args.pro_num == 3:
            self.pro_seq = nn.Sequential(nn.Linear(self.shared_dim, self.latent_dim), nn.ReLU(inplace=True),
                                         nn.Linear(self.latent_dim, self.latent_dim)).to(device)
            self.pro_gnn = nn.Sequential(nn.Linear(self.shared_dim, self.latent_dim), nn.ReLU(inplace=True),
                                         nn.Linear(self.latent_dim, self.latent_dim)).to(device)
            self.pro_geo = nn.Sequential(nn.Linear(self.shared_dim, self.latent_dim), nn.ReLU(inplace=True),
                                         nn.Linear(self.latent_dim, self.latent_dim)).to(device)
        elif args.pro_num == 1:
            self.pro_seq = nn.Sequential(nn.Linear(self.shared_dim, self.latent_dim), nn.ReLU(inplace=True),
                                         nn.Linear(self.latent_dim, self.latent_dim)).to(device)
            self.pro_gnn = self.pro_seq
            self.pro_geo = self.pro_seq

        self.entropy = loss_type[args.task_type]

        if args.pool_type == 'mean':
            self.pool = global_mean_pool
        else:
            self.pool = Global_Attention(args.seq_hidden_dim).to(self.device)

        fusion_dim = args.gnn_hidden_dim * self.graph + args.seq_hidden_dim * self.sequence + \
                     args.geo_hidden_dim * self.geometry
        if self.args.fusion == 3:
            fusion_dim = fusion_dim // (self.graph + self.sequence + self.geometry)
            self.fusion = DoubleWeightFusion(self.graph + self.sequence + self.geometry, self.shared_dim, self.shared_dim, device=self.device)
        elif self.args.fusion == 2 or self.args.fusion == 0:
            fusion_dim = args.seq_hidden_dim

        self.dropout = nn.Dropout(args.dropout)
        self.output_layer = nn.Sequential(
            nn.Linear(int(self.shared_dim), int(self.shared_dim)//2), nn.ReLU(), nn.Dropout(args.dropout),
            nn.Linear(int(self.shared_dim)//2, int(self.shared_dim)//2), nn.ReLU(), nn.Dropout(args.dropout),
            nn.Linear(int(self.shared_dim //2), args.output_dim)
        ).to(self.device)

    def forward(self, trans_batch_seq, seq_mask, batch_mask_seq, gnn_batch_graph, gnn_feature_batch, batch_mask_gnn,
                graph_dict, node_id_all, edge_id_all):
        shared_list, private_list = [], []
        mu_shared_list, logvar_shared_list = [], []
        mu_private_list, logvar_private_list = [], []
        recon_list, orig_list = [], []

        if self.graph:
            node_gnn_x = self.gnn(gnn_batch_graph, gnn_feature_batch, batch_mask_gnn)
            graph_gnn_x = self.pool(node_gnn_x, batch_mask_gnn)
            z_shared, z_private, recon, mu_s, logvar_s, mu_p, logvar_p = self.gnn_ae(graph_gnn_x)
            shared_list.append(z_shared)
            private_list.append(z_private)
            mu_shared_list.append(mu_s)
            logvar_shared_list.append(logvar_s)
            mu_private_list.append(mu_p)
            logvar_private_list.append(logvar_p)
            recon_list.append(recon)
            orig_list.append(graph_gnn_x)
        if self.sequence:
            nloss, node_seq_x = self.transformer(trans_batch_seq)
            graph_seq_x = self.pool(node_seq_x[seq_mask], batch_mask_seq)
            z_shared, z_private, recon, mu_s, logvar_s, mu_p, logvar_p = self.seq_ae(graph_seq_x)
            shared_list.append(z_shared)
            private_list.append(z_private)
            mu_shared_list.append(mu_s)
            logvar_shared_list.append(logvar_s)
            mu_private_list.append(mu_p)
            logvar_private_list.append(logvar_p)
            recon_list.append(recon)
            orig_list.append(graph_seq_x)
        if self.geometry:
            node_repr, edge_repr = self.compound_encoder(graph_dict[0], graph_dict[1], node_id_all, edge_id_all)
            graph_geo_x = self.pool(node_repr, node_id_all[0])
            z_shared, z_private, recon, mu_s, logvar_s, mu_p, logvar_p = self.geo_ae(graph_geo_x)
            shared_list.append(z_shared)
            private_list.append(z_private)
            mu_shared_list.append(mu_s)
            logvar_shared_list.append(logvar_s)
            mu_private_list.append(mu_p)
            logvar_private_list.append(logvar_p)
            recon_list.append(recon)
            orig_list.append(graph_geo_x)

        molecule_emb = self.fusion(shared_list)
        molecule_emb = self.dropout(molecule_emb)
        preds = self.output_layer(molecule_emb)
        # NaN检查，便于调试
        if torch.isnan(preds).any():
            raise ValueError("NaN detected in preds!")
        return shared_list, private_list, preds, recon_list, orig_list, mu_shared_list, logvar_shared_list, mu_private_list, logvar_private_list

    def label_loss(self, pred, label, mask):
        loss_mat = self.entropy(pred, label)
        return loss_mat.sum() / mask.sum()

    def ae_loss(self, recon_list, orig_list):
        recon_loss = sum([F.mse_loss(recon, orig).mean() for recon, orig in zip(recon_list, orig_list)]) / len(recon_list)
        return recon_loss

    def kl_loss(self, mu_list, logvar_list):
        kl = sum([-0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / mu.size(0)
                  for mu, logvar in zip(mu_list, logvar_list)]) / len(mu_list)
        return kl
    
    # def ortho_loss(self, z_shared_list, z_private_list):
    #     # 对每个模态，计算共享和私有隐变量的相关性
    #     loss = 0
    #     for z_s, z_p in zip(z_shared_list, z_private_list):
    #         # [batch, dim]
    #         # 归一化后做内积
    #         z_s = F.normalize(z_s, dim=1)
    #         z_p = F.normalize(z_p, dim=1)
    #         loss += (z_s * z_p).sum(dim=1).pow(2).mean()
    #     return loss / len(z_shared_list)
    
    # def ortho_loss(self, z_shared_list, z_private_list):
    #     loss = 0
    #     for z_s, z_p in zip(z_shared_list, z_private_list):
    #         z_s = F.normalize(z_s, dim=1)
    #         z_p = F.normalize(z_p, dim=1)
    #         corr = (z_s * z_p).sum(dim=1) / (z_s.norm(dim=1) * z_p.norm(dim=1) + 1e-8)
    #         loss += corr.pow(2).mean()
    #     return loss / len(z_shared_list)
    def rbf_kernel(self, x, sigma=1.0):
        x_norm = (x ** 2).sum(dim=1).view(-1, 1)
        dist = x_norm + x_norm.t() - 2.0 * torch.mm(x, x.t())
        k = torch.exp(-dist / (2 * sigma ** 2))
        return k

    def ortho_loss(self, z_shared_list, z_private_list, sigma=1.0):
    #def hsic_loss(self, z_shared_list, z_private_list, sigma=1.0):
        loss = 0
        for z_s, z_p in zip(z_shared_list, z_private_list):
            K = self.rbf_kernel(z_s)
            L = self.rbf_kernel(z_p)
            n = K.size(0)
            H = torch.eye(n, device=K.device) - 1.0 / n
            Kc = torch.mm(torch.mm(H, K), H)
            Lc = torch.mm(torch.mm(H, L), H)
            hsic = (Kc * Lc).sum() / ((n - 1) ** 2)
            loss += hsic
        return loss / len(z_shared_list)

    # def align_loss(self, z_shared_list):
    #     # 让不同模态的共享空间尽量一致
    #     loss = 0
    #     n = len(z_shared_list)
    #     for i in range(n):
    #         for j in range(i+1, n):
    #             loss += F.mse_loss(z_shared_list[i], z_shared_list[j])
    #     return loss / (n * (n-1) / 2)

    def align_loss(self, z_shared_list):
        loss = 0
        n = len(z_shared_list)
        for i in range(n):
            for j in range(i+1, n):
                cos_sim = F.cosine_similarity(z_shared_list[i], z_shared_list[j], dim=1)
                loss += (1 - cos_sim).mean()
        return loss / (n * (n-1) / 2)

    def loss_cal(self, preds, targets, mask, recon_list, orig_list, mu_shared_list, logvar_shared_list, mu_private_list, logvar_private_list,
                z_shared_list, z_private_list,
                recon_weight=0.2, beta_shared=0.01, beta_private=0.01, gamma_ortho=0.01, gamma_align=0.01):
        loss_label = self.label_loss(preds, targets, mask)
        recon_loss = self.ae_loss(recon_list, orig_list)
        kl_shared = self.kl_loss(mu_shared_list, logvar_shared_list)
        kl_private = self.kl_loss(mu_private_list, logvar_private_list)
        ortho = self.ortho_loss(z_shared_list, z_private_list)
        align = self.align_loss(z_shared_list)
        total_loss = loss_label + recon_weight * recon_loss + beta_shared * kl_shared + beta_private * kl_private \
                    + gamma_ortho * ortho + gamma_align * align
        # print(f"Total Loss: {total_loss.item():.4f}, Label Loss: {loss_label.item():.4f}, "
        #       f"Recon Loss: {recon_weight * recon_loss.item():.4f}, KL Shared: {beta_shared * kl_shared.item():.4f}, "
        #       f"KL Private: {beta_private * kl_private.item():.4f}, Ortho Loss: {gamma_ortho * ortho.item():.8f}, Align Loss: {gamma_align * align.item():.8f}")
        return total_loss, loss_label, recon_loss#, ortho, align