import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.nn import global_mean_pool, GlobalAttention
from torch.nn import init
from torch.nn.parameter import Parameter
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

class GatedAttentionFusion(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_modalities):
        super().__init__()
        self.attention_network = nn.Sequential(
            nn.Linear(input_dim * num_modalities, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, num_modalities),
            nn.Softmax(dim=1)
        )
        self.norm = nn.LayerNorm(input_dim)

    def forward(self, features_list: list) -> Tensor:
        concatenated_features = torch.cat(features_list, dim=1)
        attention_weights = self.attention_network(concatenated_features).unsqueeze(2)
        stacked_features = torch.stack(features_list, dim=1)
        fused_features = (stacked_features * attention_weights).sum(dim=1)
        residual = torch.mean(stacked_features, dim=1)
        output = self.norm(fused_features + residual)
        return output

class VIBHead(nn.Module):
    def __init__(
        self,
        in_dim,
        shared_dim,
        private_dim,
        hidden_dim=384,
        dropout=0.5,
        norm=True
    ):
        super().__init__()
        encoder_backbone = [
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
        ]
        if norm:
            encoder_backbone.append(nn.LayerNorm(hidden_dim))
        self.encoder_backbone = nn.Sequential(*encoder_backbone)
        self.dropout = nn.Dropout(dropout)
        self.shared_head = nn.Linear(hidden_dim, hidden_dim // 2)
        self.private_head = nn.Linear(hidden_dim, hidden_dim // 2)
        self.fc_mu_shared = nn.Linear(hidden_dim // 2, shared_dim)
        self.fc_logvar_shared = nn.Linear(hidden_dim // 2, shared_dim)
        self.fc_mu_private = nn.Linear(hidden_dim // 2, private_dim)
        self.fc_logvar_private = nn.Linear(hidden_dim // 2, private_dim)

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        else:
            return mu

    def forward(self, x):
        h = self.encoder_backbone(x)
        h_shared = self.dropout(self.shared_head(h))
        h_private = self.dropout(self.private_head(h))
        mu_shared = self.fc_mu_shared(h_shared)
        logvar_shared = torch.clamp(self.fc_logvar_shared(h_shared), min=-10, max=10)
        mu_private = self.fc_mu_private(h_private)
        logvar_private = torch.clamp(self.fc_logvar_private(h_private), min=-10, max=10)
        z_shared = self.reparameterize(mu_shared, logvar_shared)
        z_private = self.reparameterize(mu_private, logvar_private)
        return z_shared, z_private, mu_shared, logvar_shared, mu_private, logvar_private

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
        # self.shared_dim = getattr(args, 'vib_shared_dim', args.latent_dim // 2)
        # self.private_dim = getattr(args, 'vib_private_dim', args.latent_dim // 2)
        self.num_modalities = args.graph + args.sequence + args.geometry

        self.shared_dim = args.vib_shared_dim
        self.private_dim = args.vib_private_dim
        self.beta_shared = args.beta_shared
        self.mmd_private_weight = args.mmd_private_weight
        self.align_weight = args.align_weight
        self.vib_hidden_dim = args.vib_hidden_dim
        self.vib_dropout = args.vib_dropout
        self.vib_norm = bool(args.vib_norm)

        # 编码器
        self.gnn = MPNEncoder(
            atom_fdim=args.gnn_atom_dim,
            bond_fdim=args.gnn_bond_dim,
            hidden_size=args.gnn_hidden_dim,
            bias=args.bias,
            depth=args.gnn_num_layers,
            dropout=args.dropout,
            activation=args.gnn_activation,
            device=device
        )
        self.transformer = TrfmSeq2seq(
            input_dim=args.seq_input_dim,
            hidden_size=args.seq_hidden_dim,
            num_head=args.seq_num_heads,
            n_layers=args.seq_num_layers,
            dropout=args.dropout,
            vocab_num=args.vocab_num,
            device=device,
            recons=args.recons
        ).to(self.device)
        self.compound_encoder = GeoGNNModel(args, compound_encoder_config, device)

        # VIB头，参数全部可配置
        self.gnn_vib = VIBHead(
            in_dim=args.gnn_hidden_dim,
            shared_dim=self.shared_dim,
            private_dim=self.private_dim,
            hidden_dim=self.vib_hidden_dim,
            dropout=self.vib_dropout,
            norm=self.vib_norm
        ).to(device)
        self.seq_vib = VIBHead(
            in_dim=args.seq_hidden_dim,
            shared_dim=self.shared_dim,
            private_dim=self.private_dim,
            hidden_dim=self.vib_hidden_dim,
            dropout=self.vib_dropout,
            norm=self.vib_norm
        ).to(device)
        self.geo_vib = VIBHead(
            in_dim=args.geo_hidden_dim,
            shared_dim=self.shared_dim,
            private_dim=self.private_dim,
            hidden_dim=self.vib_hidden_dim,
            dropout=self.vib_dropout,
            norm=self.vib_norm
        ).to(device)

        self.task_loss_fn = loss_type[args.task_type]
        self.pool = global_mean_pool
        self.fusion = GatedAttentionFusion(self.shared_dim, self.shared_dim // 2, self.num_modalities).to(self.device)
        self.dropout = nn.Dropout(args.dropout).to(self.device)
        
        self.output_layer = nn.Sequential(
            nn.Linear(self.shared_dim, self.shared_dim // 2), nn.ReLU(),
            nn.Dropout(args.dropout),
            nn.Linear(self.shared_dim // 2, args.output_dim)
        ).to(self.device)

    def forward(self, trans_batch_seq, seq_mask, batch_mask_seq, gnn_batch_graph, gnn_feature_batch, batch_mask_gnn,
                graph_dict, node_id_all, edge_id_all):
        shared_list, private_list = [], []
        mu_shared_list, logvar_shared_list = [], []
        mu_private_list, logvar_private_list = [], []
        
        if self.graph:
            node_gnn_x = self.gnn(gnn_batch_graph, gnn_feature_batch, batch_mask_gnn)
            graph_gnn_x = self.pool(node_gnn_x, batch_mask_gnn)
            z_s, z_p, mu_s, logvar_s, mu_p, logvar_p = self.gnn_vib(graph_gnn_x)
            shared_list.append(F.normalize(z_s, p=2, dim=1) if self.args.norm else z_s)
            private_list.append(z_p)
            mu_shared_list.append(mu_s); logvar_shared_list.append(logvar_s)
            mu_private_list.append(mu_p); logvar_private_list.append(logvar_p)

        if self.sequence:
            nloss, node_seq_x = self.transformer(trans_batch_seq)
            graph_seq_x = self.pool(node_seq_x[seq_mask], batch_mask_seq)
            z_s, z_p, mu_s, logvar_s, mu_p, logvar_p = self.seq_vib(graph_seq_x)
            shared_list.append(F.normalize(z_s, p=2, dim=1) if self.args.norm else z_s)
            private_list.append(z_p)
            mu_shared_list.append(mu_s); logvar_shared_list.append(logvar_s)
            mu_private_list.append(mu_p); logvar_private_list.append(logvar_p)

        if self.geometry:
            node_repr, edge_repr = self.compound_encoder(graph_dict[0], graph_dict[1], node_id_all, edge_id_all)
            graph_geo_x = self.pool(node_repr, node_id_all[0])
            z_s, z_p, mu_s, logvar_s, mu_p, logvar_p = self.geo_vib(graph_geo_x)
            shared_list.append(F.normalize(z_s, p=2, dim=1) if self.args.norm else z_s)
            private_list.append(z_p)
            mu_shared_list.append(mu_s); logvar_shared_list.append(logvar_s)
            mu_private_list.append(mu_p); logvar_private_list.append(logvar_p)

        if self.args.fusion == 1:
            molecule_emb = torch.cat(shared_list, dim=1)
        elif self.args.fusion == 2:
            molecule_emb = shared_list[0]
            for i in range(1, len(shared_list)):
                molecule_emb = molecule_emb.mul(shared_list[i])
        elif self.args.fusion == 3:
            molecule_emb = self.fusion(shared_list)
        else:
            molecule_emb = torch.mean(torch.stack(shared_list, dim=0), dim=0)

        if not self.args.norm:
            molecule_emb = self.dropout(molecule_emb)

        pred = self.output_layer(molecule_emb)
        
        return shared_list, private_list, pred, mu_shared_list, logvar_shared_list, mu_private_list, logvar_private_list

    def label_loss(self, pred, label, mask):
        loss_mat = self.task_loss_fn(pred, label)
        loss_mat = loss_mat * mask
        return loss_mat.sum() / (mask.sum() + 1e-8)

    def kl_loss(self, mu: Tensor, logvar: Tensor) -> Tensor:
        return -0.5 * torch.mean(torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1))

    def mmd_loss(self, x, y, kernel="rbf"):
        def rbf_kernel(x, y, gamma=1.0):
            x_size, y_size = x.size(0), y.size(0)
            dim = x.size(1)
            x = x.unsqueeze(1)
            y = y.unsqueeze(0)
            tiled_x = x.expand(x_size, y_size, dim)
            tiled_y = y.expand(x_size, y_size, dim)
            return torch.exp(-gamma * (tiled_x - tiled_y).pow(2).mean(2))

        xx = rbf_kernel(x, x)
        yy = rbf_kernel(y, y)
        xy = rbf_kernel(x, y)
        return xx.mean() + yy.mean() - 2 * xy.mean()
    
    def cl_loss(self, x1, x2, T=0.1):
        batch_size, _ = x1.size()
        x1_abs = x1.norm(dim=1)
        x2_abs = x2.norm(dim=1)
        sim_matrix = torch.einsum('ik,jk->ij', x1, x2) / (torch.einsum('i,j->ij', x1_abs, x2_abs) + 1e-8)
        sim_matrix = torch.exp(sim_matrix / T)
        pos_sim = sim_matrix[range(batch_size), range(batch_size)]
        loss = -torch.log(pos_sim / (sim_matrix.sum(dim=1) - pos_sim + 1e-8)).mean()
        return loss

    def loss_cal(self, preds, targets, mask, 
                     mu_shared_list, logvar_shared_list, mu_private_list, logvar_private_list,
                     z_shared_list, z_private_list):
        
        # 1. 主要任务损失
        loss_label = self.label_loss(preds, targets, mask)
        
        # 2. 共享潜变量的KL损失
        kl_shared = sum([self.kl_loss(mu, logvar) for mu, logvar in zip(mu_shared_list, logvar_shared_list)])
        if self.num_modalities > 0:
            kl_shared /= self.num_modalities
            
        # 3. 私有潜变量的MMD损失
        mmd_private = torch.tensor(0.0, device=self.device)
        if z_private_list:
            prior_dist = torch.randn_like(z_private_list[0])
            mmd_private = sum([self.mmd_loss(z_p, prior_dist) for z_p in z_private_list])
            if self.num_modalities > 0:
                mmd_private /= self.num_modalities
        
        # 4. 对比对齐损失
        align = torch.tensor(0.0, dtype=torch.float, device=self.device)
        if len(z_shared_list) > 1:
            num_pairs = 0
            for i in range(len(z_shared_list)):
                for j in range(i + 1, len(z_shared_list)):
                    align += self.cl_loss(z_shared_list[i], z_shared_list[j])
                    num_pairs += 1
            if num_pairs > 0:
                align /= num_pairs
        
        aux_loss = (
            self.beta_shared * kl_shared +
            self.mmd_private_weight * mmd_private +
            self.align_weight * align
        )
        
        total_loss = loss_label + aux_loss
        
        return total_loss, loss_label, aux_loss