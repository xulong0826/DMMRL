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

class GatedAttentionFusion(nn.Module):
    def __init__(
        self, num_modalities, shared_dim=192, out_dim=192, num_heads=4, dropout=0.1, bias=True,
        device=None, dtype=None, residual=True, use_layernorm=True, debug_nan=False
    ):
        super().__init__()
        self.num_modalities = num_modalities
        self.shared_dim = shared_dim
        self.out_dim = out_dim
        self.num_heads = num_heads
        self.residual = residual
        self.debug_nan = debug_nan

        factory_kwargs = {'device': device, 'dtype': dtype}
        self.attn = nn.MultiheadAttention(embed_dim=shared_dim, num_heads=num_heads, dropout=dropout, batch_first=True, **factory_kwargs)
        self.proj = nn.Linear(shared_dim, out_dim, bias=bias, **factory_kwargs)
        self.dropout = nn.Dropout(dropout)
        self.gate = nn.Parameter(torch.full((num_modalities,), 0.1, **{k: v for k, v in factory_kwargs.items() if v is not None}))
        self.use_layernorm = use_layernorm
        if use_layernorm:
            self.ln = nn.LayerNorm(out_dim, **factory_kwargs)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.proj.weight)
        if self.proj.bias is not None:
            nn.init.zeros_(self.proj.bias)
        nn.init.xavier_uniform_(self.attn.in_proj_weight)
        nn.init.zeros_(self.attn.in_proj_bias)
        nn.init.constant_(self.gate, 0.1)
        if hasattr(self, "ln"):
            self.ln.reset_parameters()

    def forward(self, shared_list):
        x = torch.stack(shared_list, dim=1)  # [batch, num_modalities, shared_dim]
        device, dtype = x.device, x.dtype
        attn_out, _ = self.attn(x, x, x)
        gate_weights = torch.softmax(self.gate, dim=0).to(device=device, dtype=dtype).view(1, -1, 1)
        fused = (attn_out * gate_weights).sum(dim=1)
        if self.residual:
            fused = fused + x.mean(dim=1)
        fused = self.dropout(fused)
        fused = self.proj(fused)
        if self.use_layernorm:
            fused = self.ln(fused)
        if self.debug_nan and torch.isnan(fused).any():
            raise ValueError("NaN detected in GatedAttentionFusion output!")
        return fused

class VAEHead(nn.Module):
    def __init__(self, in_dim, shared_dim, private_dim, hidden_dim=512, dropout=0.1, out_act=None, norm=True):
        super().__init__()
        encoder_layers = [
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
        ]
        if norm:
            encoder_layers.append(nn.LayerNorm(hidden_dim))
        encoder_layers.append(nn.Dropout(dropout))
        self.encoder_shared = nn.Sequential(*encoder_layers)
        self.encoder_private = nn.Sequential(*encoder_layers)

        self.fc_mu_shared = nn.Linear(hidden_dim, shared_dim)
        self.fc_logvar_shared = nn.Linear(hidden_dim, shared_dim)
        self.fc_mu_private = nn.Linear(hidden_dim, private_dim)
        self.fc_logvar_private = nn.Linear(hidden_dim, private_dim)

        decoder_layers = [
            nn.Linear(shared_dim + private_dim, hidden_dim),
            nn.ReLU(),
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
        assert not torch.isnan(x).any(), "NaN in VAE input x"
        h_shared = self.encoder_shared(x)
        h_private = self.encoder_private(x)
        # assert not torch.isnan(h).any(), "NaN in VAE encoder output h"
        mu_shared = self.fc_mu_shared(h_shared)
        logvar_shared = torch.clamp(self.fc_logvar_shared(h_shared), min=-10, max=10)
        mu_private = self.fc_mu_private(h_private)
        logvar_private = torch.clamp(self.fc_logvar_private(h_private), min=-10, max=10)
        assert not torch.isnan(mu_shared).any(), "NaN in mu_shared"
        assert not torch.isnan(logvar_shared).any(), "NaN in logvar_shared"
        assert not torch.isnan(mu_private).any(), "NaN in mu_private"
        assert not torch.isnan(logvar_private).any(), "NaN in logvar_private"
        z_shared = self.reparameterize(mu_shared, logvar_shared)
        z_private = self.reparameterize(mu_private, logvar_private)
        z = torch.cat([z_shared, z_private], dim=1)
        recon_x = self.decoder(z)
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

        # 投影私有空间到共享空间维度，便于正交损失
        self.private2shared_proj = nn.Linear(self.private_dim, self.shared_dim).to(device)

        if args.pro_num == 3:
            self.pro_seq = nn.Sequential(
                nn.Linear(self.shared_dim, self.latent_dim), nn.ReLU(inplace=True),
                nn.LayerNorm(self.latent_dim),  # 统一加LayerNorm
                nn.Linear(self.latent_dim, self.latent_dim), nn.ReLU(inplace=True),
                nn.LayerNorm(self.latent_dim)
            ).to(device)
            self.pro_gnn = nn.Sequential(
                nn.Linear(self.shared_dim, self.latent_dim), nn.ReLU(inplace=True),
                nn.LayerNorm(self.latent_dim),
                nn.Linear(self.latent_dim, self.latent_dim), nn.ReLU(inplace=True),
                nn.LayerNorm(self.latent_dim)
            ).to(device)
            self.pro_geo = nn.Sequential(
                nn.Linear(self.shared_dim, self.latent_dim), nn.ReLU(inplace=True),
                nn.LayerNorm(self.latent_dim),
                nn.Linear(self.latent_dim, self.latent_dim), nn.ReLU(inplace=True),
                nn.LayerNorm(self.latent_dim)
            ).to(device)
        elif args.pro_num == 1:
            self.pro_seq = nn.Sequential(
                nn.Linear(self.shared_dim, self.latent_dim), nn.ReLU(inplace=True),
                nn.LayerNorm(self.latent_dim),
                nn.Linear(self.latent_dim, self.latent_dim), nn.ReLU(inplace=True),
                nn.LayerNorm(self.latent_dim)
            ).to(device)
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
            self.fusion = GatedAttentionFusion(self.graph + self.sequence + self.geometry, self.shared_dim, self.shared_dim, device=self.device)
        elif self.args.fusion == 2 or self.args.fusion == 0:
            fusion_dim = args.seq_hidden_dim

        self.dropout = nn.Dropout(args.dropout)
        self.output_layer = nn.Sequential(
            nn.Linear(self.shared_dim, self.shared_dim//2), nn.ReLU(),
            nn.LayerNorm(self.shared_dim//2),
            nn.Linear(self.shared_dim//2, args.output_dim)
        ).to(self.device)

    def forward(self, trans_batch_seq, seq_mask, batch_mask_seq, gnn_batch_graph, gnn_feature_batch, batch_mask_gnn,
                graph_dict, node_id_all, edge_id_all):
        shared_list, private_list = [], []
        mu_shared_list, logvar_shared_list = [], []
        mu_private_list, logvar_private_list = [], []
        recon_list, orig_list = [], []

        if self.graph:
            node_gnn_x = self.gnn(gnn_batch_graph, gnn_feature_batch, batch_mask_gnn)
            node_gnn_x = F.layer_norm(node_gnn_x, node_gnn_x.shape[-1:])  # 统一正则化
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
            node_seq_x = F.layer_norm(node_seq_x, node_seq_x.shape[-1:])
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
            node_repr = F.layer_norm(node_repr, node_repr.shape[-1:])
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

    def ortho_loss(self, z_shared_list, z_private_list):
        loss = 0
        for z_s, z_p in zip(z_shared_list, z_private_list):
            z_s = F.normalize(z_s, dim=1)
            z_p_proj = self.private2shared_proj(z_p)
            z_p_proj = F.normalize(z_p_proj, dim=1)
            corr = (z_s * z_p_proj).sum(dim=1)
            loss += corr.pow(2).mean()
        return loss / len(z_shared_list)

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
                recon_weight=0.2, beta_shared=0.01, beta_private=0.01, gamma_ortho=0.001, gamma_align=0.001):
        loss_label = self.label_loss(preds, targets, mask)
        recon_loss = self.ae_loss(recon_list, orig_list)
        kl_shared = self.kl_loss(mu_shared_list, logvar_shared_list)
        kl_private = self.kl_loss(mu_private_list, logvar_private_list)
        ortho = self.ortho_loss(z_shared_list, z_private_list)
        align = self.align_loss(z_shared_list)
        total_loss = loss_label + recon_weight * recon_loss + beta_shared * kl_shared + beta_private * kl_private + gamma_ortho * ortho + gamma_align * align
        # print(f"Total Loss: {total_loss.item():.4f}, Label Loss: {loss_label.item():.4f}, "
        #       f"Recon Loss: {recon_weight * recon_loss.item():.4f}, KL Shared: {beta_shared * kl_shared.item():.4f}, "
        #       f"KL Private: {beta_private * kl_private.item():.4f}, Ortho Loss: {gamma_ortho * ortho.item():.8f}, Align Loss: {gamma_align * align.item():.8f}")
        return total_loss, loss_label, recon_loss