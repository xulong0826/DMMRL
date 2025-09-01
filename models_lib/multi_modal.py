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
        norm_type="layer",  # 'layer', 'batch', 'none'
        residual=False,     # 是否使用残差连接
        depth=2,            # 编码器深度（1或2）
        clamp_logvar=True   # 是否裁剪logvar
    ):
        super().__init__()
        
        # 保存配置
        self.norm_type = norm_type
        self.residual = residual
        self.depth = depth
        self.clamp_logvar = clamp_logvar
        
        # 构建编码器骨干网络
        encoder_backbone = []
        
        # 第一层
        encoder_backbone.append(nn.Linear(in_dim, hidden_dim))
        encoder_backbone.append(nn.ReLU())
        
        # 添加归一化层（如果需要）
        if norm_type == "layer":
            encoder_backbone.append(nn.LayerNorm(hidden_dim))
        elif norm_type == "batch":
            encoder_backbone.append(nn.BatchNorm1d(hidden_dim))
            
        # 添加dropout
        encoder_backbone.append(nn.Dropout(dropout))
        
        self.encoder_backbone = nn.Sequential(*encoder_backbone)
        
        # 深度=2时的额外层
        if depth > 1:
            self.shared_head = self._create_head_layer(hidden_dim, hidden_dim // 2, 
                                                     norm_type, dropout)
            self.private_head = self._create_head_layer(hidden_dim, hidden_dim // 2, 
                                                      norm_type, dropout)
            shared_out_dim = hidden_dim // 2
            private_out_dim = hidden_dim // 2
        else:
            # 深度=1时直接从backbone输出
            self.shared_head = nn.Identity()
            self.private_head = nn.Identity()
            shared_out_dim = hidden_dim
            private_out_dim = hidden_dim
            
        # 输出层
        self.fc_mu_shared = nn.Linear(shared_out_dim, shared_dim)
        self.fc_logvar_shared = nn.Linear(shared_out_dim, shared_dim)
        self.fc_mu_private = nn.Linear(private_out_dim, private_dim)
        self.fc_logvar_private = nn.Linear(private_out_dim, private_dim)
        
        # 如果使用残差连接，需要投影层来匹配维度
        if residual and depth > 1:
            self.shared_proj = nn.Linear(hidden_dim, hidden_dim // 2)
            self.private_proj = nn.Linear(hidden_dim, hidden_dim // 2)
        
    def _create_head_layer(self, in_dim, out_dim, norm_type, dropout):
        layers = [nn.Linear(in_dim, out_dim), nn.ReLU()]
        
        if norm_type == "layer":
            layers.append(nn.LayerNorm(out_dim))
        elif norm_type == "batch":
            layers.append(nn.BatchNorm1d(out_dim))
            
        layers.append(nn.Dropout(dropout))
        return nn.Sequential(*layers)
        
    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        else:
            return mu

    def forward(self, x):
        # 编码器骨干网络
        h = self.encoder_backbone(x)
        
        # 共享表示路径
        if self.residual and self.depth > 1:
            h_shared_proj = self.shared_proj(h)
            h_shared = self.shared_head(h)
            h_shared = h_shared + h_shared_proj
        else:
            h_shared = self.shared_head(h)
            
        # 私有表示路径
        if self.residual and self.depth > 1:
            h_private_proj = self.private_proj(h)
            h_private = self.private_head(h)
            h_private = h_private + h_private_proj
        else:
            h_private = self.private_head(h)
        
        # 共享表示的均值和方差
        mu_shared = self.fc_mu_shared(h_shared)
        logvar_shared = self.fc_logvar_shared(h_shared)
        if self.clamp_logvar:
            logvar_shared = torch.clamp(logvar_shared, min=-10, max=10)
            
        # 私有表示的均值和方差
        mu_private = self.fc_mu_private(h_private)
        logvar_private = self.fc_logvar_private(h_private)
        if self.clamp_logvar:
            logvar_private = torch.clamp(logvar_private, min=-10, max=10)
            
        # 重参数化采样
        z_shared = self.reparameterize(mu_shared, logvar_shared)
        z_private = self.reparameterize(mu_private, logvar_private)
        
        return z_shared, z_private, mu_shared, logvar_shared, mu_private, logvar_private

class SimpleDecoder(nn.Module):
    """简单高效的解码器，保持模型轻量级"""
    def __init__(self, in_dim, out_dim, hidden_dim=None, use_norm=False):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = (in_dim + out_dim) // 2  # 自动计算中间层大小
        
        layers = [
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU()
        ]
        
        # 可选的归一化层
        if use_norm:
            layers.append(nn.LayerNorm(hidden_dim))
            
        layers.append(nn.Linear(hidden_dim, out_dim))
        
        self.decoder = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.decoder(x)

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
        self.num_modalities = args.graph + args.sequence + args.geometry

        # VIB相关参数
        self.vib_norm = bool(args.vib_norm)
        self.warmup_epochs = args.kl_warmup_epochs
        self.shared_dim = args.vib_shared_dim
        self.private_dim = args.vib_private_dim
        self.beta_shared = args.beta_shared
        self.mmd_private_weight = args.mmd_private_weight
        self.align_weight = args.align_weight
        self.vib_hidden_dim = args.vib_hidden_dim
        self.vib_dropout = args.vib_dropout
        self.ortho_weight = args.ortho_weight
        self.recon_weight = args.recon_weight
        
        # 新增参数读取
        self.vib_norm_type = getattr(args, 'vib_norm_type', 'layer')
        self.vib_residual = getattr(args, 'vib_residual', False)
        self.vib_depth = getattr(args, 'vib_depth', 2)
        self.vib_clamp_logvar = getattr(args, 'vib_clamp_logvar', True)

        # 更新VIB头的初始化
        self.gnn_vib = VIBHead(
            in_dim=args.gnn_hidden_dim,
            shared_dim=self.shared_dim,
            private_dim=self.private_dim,
            hidden_dim=self.vib_hidden_dim,
            dropout=self.vib_dropout,
            norm_type=self.vib_norm_type,
            residual=self.vib_residual,
            depth=self.vib_depth,
            clamp_logvar=self.vib_clamp_logvar
        ).to(device)
        
        self.seq_vib = VIBHead(
            in_dim=args.seq_hidden_dim,
            shared_dim=self.shared_dim,
            private_dim=self.private_dim,
            hidden_dim=self.vib_hidden_dim,
            dropout=self.vib_dropout,
            norm_type=self.vib_norm_type,
            residual=self.vib_residual,
            depth=self.vib_depth,
            clamp_logvar=self.vib_clamp_logvar
        ).to(device)
        
        self.geo_vib = VIBHead(
            in_dim=args.geo_hidden_dim,
            shared_dim=self.shared_dim,
            private_dim=self.private_dim,
            hidden_dim=self.vib_hidden_dim,
            dropout=self.vib_dropout,
            norm_type=self.vib_norm_type,
            residual=self.vib_residual,
            depth=self.vib_depth,
            clamp_logvar=self.vib_clamp_logvar
        ).to(device)

        # 简单参数
        use_decoder_norm = self.vib_norm        
        if self.sequence:
            self.seq_decoder = SimpleDecoder(
                in_dim=self.shared_dim + self.private_dim,
                out_dim=args.seq_hidden_dim,
                hidden_dim=self.vib_hidden_dim,
                use_norm=use_decoder_norm
            ).to(device)

        if self.graph:
            self.gnn_decoder = SimpleDecoder(
                in_dim=self.shared_dim + self.private_dim,
                out_dim=args.gnn_hidden_dim,
                hidden_dim=self.vib_hidden_dim,
                use_norm=use_decoder_norm
            ).to(device)

        if self.geometry:
            self.geo_decoder = SimpleDecoder(
                in_dim=self.shared_dim + self.private_dim,
                out_dim=args.geo_hidden_dim,
                hidden_dim=self.vib_hidden_dim,
                use_norm=use_decoder_norm
            ).to(device)

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
        original_features = []
        recon_features = []
        
        if self.graph:
            node_gnn_x = self.gnn(gnn_batch_graph, gnn_feature_batch, batch_mask_gnn)
            graph_gnn_x = self.pool(node_gnn_x, batch_mask_gnn)
            original_features.append(graph_gnn_x)
            
            z_s, z_p, mu_s, logvar_s, mu_p, logvar_p = self.gnn_vib(graph_gnn_x)
            shared_list.append(F.normalize(z_s, p=2, dim=1) if self.args.norm else z_s)
            private_list.append(z_p)
            mu_shared_list.append(mu_s); logvar_shared_list.append(logvar_s)
            mu_private_list.append(mu_p); logvar_private_list.append(logvar_p)
            
            # Reconstruction
            recon_input = torch.cat([z_s, z_p], dim=1)
            recon_gnn = self.gnn_decoder(recon_input)
            recon_features.append(recon_gnn)

        if self.sequence:
            nloss, node_seq_x = self.transformer(trans_batch_seq)
            graph_seq_x = self.pool(node_seq_x[seq_mask], batch_mask_seq)
            original_features.append(graph_seq_x)
            
            z_s, z_p, mu_s, logvar_s, mu_p, logvar_p = self.seq_vib(graph_seq_x)
            shared_list.append(F.normalize(z_s, p=2, dim=1) if self.args.norm else z_s)
            private_list.append(z_p)
            mu_shared_list.append(mu_s); logvar_shared_list.append(logvar_s)
            mu_private_list.append(mu_p); logvar_private_list.append(logvar_p)
            
            # Reconstruction
            recon_input = torch.cat([z_s, z_p], dim=1)
            recon_seq = self.seq_decoder(recon_input)
            recon_features.append(recon_seq)

        if self.geometry:
                node_repr, edge_repr = self.compound_encoder(graph_dict[0], graph_dict[1], node_id_all, edge_id_all)
                graph_geo_x = self.pool(node_repr, node_id_all[0])
                original_features.append(graph_geo_x)
                
                z_s, z_p, mu_s, logvar_s, mu_p, logvar_p = self.geo_vib(graph_geo_x)
                shared_list.append(F.normalize(z_s, p=2, dim=1) if self.args.norm else z_s)
                private_list.append(z_p)
                mu_shared_list.append(mu_s); logvar_shared_list.append(logvar_s)
                mu_private_list.append(mu_p); logvar_private_list.append(logvar_p)
                
                # 移入geometry条件内部执行重构
                recon_input = torch.cat([z_s, z_p], dim=1)
                recon_geo = self.geo_decoder(recon_input)
                recon_features.append(recon_geo)

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
        
        # return shared_list, private_list, pred, mu_shared_list, logvar_shared_list, mu_private_list, logvar_private_list
        return shared_list, private_list, pred, mu_shared_list, logvar_shared_list, mu_private_list, logvar_private_list, original_features, recon_features

    def label_loss(self, pred, label, mask):
        loss_mat = self.task_loss_fn(pred, label)
        loss_mat = loss_mat * mask
        return loss_mat.sum() / (mask.sum() + 1e-8)

    def kl_loss(self, mu: Tensor, logvar: Tensor) -> Tensor:
        # 按潜空间维度归一化
        dim = mu.size(1)
        return -0.5 * torch.mean(torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)) / dim

    def mmd_loss(self, x, y, gamma=None, normalize=True):
        # 保证输入的正确性
        x = x.float()
        y = y.float()
        assert x.dim() == 2, f"x must be 2D tensor ([batch, dim]), got {x.shape}"
        assert y.dim() == 2, f"y must be 2D tensor ([batch, dim]), got {y.shape}"
        assert torch.isfinite(x).all(), "x contains NaN or Inf"
        assert torch.isfinite(y).all(), "y contains NaN or Inf"
    
        if gamma is None:
            dists = torch.cdist(x, y, p=2)
            gamma = 1.0 / (dists.mean().item() + 1e-8)
        K_xx = torch.exp(-gamma * torch.cdist(x, x, p=2).pow(2))
        K_yy = torch.exp(-gamma * torch.cdist(y, y, p=2).pow(2))
        K_xy = torch.exp(-gamma * torch.cdist(x, y, p=2).pow(2))
        mmd = K_xx.mean() + K_yy.mean() - 2 * K_xy.mean()
        if normalize:
            mmd = mmd / x.size(1)
        return mmd
    
    # def cl_loss(self, x1, x2, T=0.1):
    #     batch_size, _ = x1.size()
    #     x1_abs = x1.norm(dim=1)
    #     x2_abs = x2.norm(dim=1)
    #     sim_matrix = torch.einsum('ik,jk->ij', x1, x2) / (torch.einsum('i,j->ij', x1_abs, x2_abs) + 1e-8)
    #     sim_matrix = torch.exp(sim_matrix / T)
    #     pos_sim = sim_matrix[range(batch_size), range(batch_size)]
    #     loss = -torch.log(pos_sim / (sim_matrix.sum(dim=1) - pos_sim + 1e-8)).mean()
    #     return loss
    def cl_loss(self, x1, x2, T=0.1):
        x1_norm, x2_norm = F.normalize(x1, p=2, dim=1), F.normalize(x2, p=2, dim=1)
        sim = torch.mm(x1_norm, x2_norm.t()) / T
        loss = -torch.mean(torch.diag(F.log_softmax(sim, dim=1)))
        return loss
    
    def orthogonality_loss(self, shared, private):
        # 归一化后取均值
        shared_norm = F.normalize(shared, p=2, dim=1)
        private_norm = F.normalize(private, p=2, dim=1)
        cosine_sim = torch.sum(shared_norm * private_norm, dim=1)
        return torch.mean(torch.abs(cosine_sim))

    def loss_cal(self, epoch, preds, targets, mask, 
                mu_shared_list, logvar_shared_list, mu_private_list, logvar_private_list,
                z_shared_list, z_private_list, original_features=None, recon_features=None):
        
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
        
        # 5. 正交损失 - 让共享和私有表示相互正交
        ortho = torch.tensor(0.0, device=self.device)
        if z_shared_list and z_private_list:
            ortho = sum([self.orthogonality_loss(z_s, z_p) 
                        for z_s, z_p in zip(z_shared_list, z_private_list)])
            ortho /= len(z_shared_list)
        
        # 6. 重构损失
        recon = torch.tensor(0.0, device=self.device)
        if original_features and recon_features:
            recon = sum([F.mse_loss(orig, recon, reduction='mean') / orig.size(1)
                         for orig, recon in zip(original_features, recon_features)])
            recon /= len(original_features)
        
        kl_factor = 1.0 / (1.0 + math.exp(-10 * (epoch / self.warmup_epochs - 0.5)))  # sigmoid递增
        other_factor = 1.0 / (1.0 + math.exp(-10 * (epoch / (self.warmup_epochs / 2 ) - 0.5)))
        aux_loss = (
            self.beta_shared * kl_shared * kl_factor +
            self.mmd_private_weight * mmd_private * other_factor +
            self.align_weight * align * other_factor +
            self.ortho_weight * ortho * kl_factor +
            self.recon_weight * recon * other_factor
        )
        total_loss = (
            loss_label + 
            self.beta_shared * kl_shared * kl_factor +
            self.mmd_private_weight * mmd_private * other_factor +
            self.align_weight * align * other_factor +
            self.ortho_weight * ortho * kl_factor +
            self.recon_weight * recon * other_factor
        )
        
        # print(f"loss_label: {loss_label:.6f}")
        # print(f"kl_shared: {kl_shared:.6f}")
        # print(f"mmd_private: {mmd_private:.6f}")
        # print(f"align: {align:.6f}")
        # print(f"ortho: {ortho:.6f}")
        # print(f"recon: {recon:.6f}")
        
        return total_loss, loss_label, aux_loss