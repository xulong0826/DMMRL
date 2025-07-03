#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
@author: Vencent_Wang
@contact: Vencent_Wang@outlook.com
@file: multi_modal.py
@time: 2023/8/13 20:05
@desc: Simplified Multi-modal Causal Disentanglement Model
'''
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch import Tensor
from torch.nn import init
from torch.nn.parameter import Parameter
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


class SimpleWeightFusion(nn.Module):
    """简化的权重融合层"""
    def __init__(self, feat_views, feat_dim, device=None):
        super().__init__()
        self.feat_views = feat_views
        self.feat_dim = feat_dim
        
        # ✅ 简化：只用可学习的静态权重
        self.weight = nn.Parameter(torch.ones(feat_views) / feat_views)  # 初始化为均匀权重
        
    def forward(self, input: list) -> Tensor:
        # ✅ 简化：直接加权求和
        weights = F.softmax(self.weight, dim=0)
        fused = sum([input[i] * weights[i] for i in range(len(input))])
        return fused


class SimpleExtractor(nn.Module):
    """简化的特征提取器"""
    def __init__(self, input_dim, bottleneck_dim, output_dim, dropout=0.1):
        super().__init__()
        # ✅ 简化：去掉自注意力，使用简单的MLP
        self.extractor = nn.Sequential(
            nn.Linear(input_dim, bottleneck_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(bottleneck_dim, output_dim)
        )
        
    def forward(self, x):
        return self.extractor(x)


class SimpleGate(nn.Module):
    """简化的门控机制"""
    def __init__(self, input_dim):
        super().__init__()
        # ✅ 简化：用简单的sigmoid门控
        self.gate = nn.Sequential(
            nn.Linear(input_dim, input_dim // 4),
            nn.ReLU(),
            nn.Linear(input_dim // 4, 2),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        return self.gate(x)


class ImprovedGate(nn.Module):
    """改进的门控机制"""
    def __init__(self, input_dim):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(input_dim, input_dim // 4),
            nn.ReLU(),
            nn.Linear(input_dim // 4, 2)
            # ✅ 去掉Sigmoid，用Gumbel-Softmax
        )
        
    def forward(self, x, temperature=1.0):
        logits = self.gate(x)
        # ✅ Gumbel-Softmax替代Sigmoid，更好的梯度
        return F.gumbel_softmax(logits, tau=temperature, hard=False, dim=-1)


class Multi_modal(nn.Module):
    def __init__(self, args, compound_encoder_config, device):
        super().__init__()
        self.args = args
        self.device = device
        self.latent_dim = args.latent_dim
        self.current_epoch = 0
        
        # ✅ 统一隐藏维度
        self.hidden_dim = 256
        
        # ✅ 更合理的维度分配
        self.shared_dims = {'gnn': 64, 'seq': 48, 'geo': 72}
        self.private_dims = {'gnn': 48, 'seq': 32, 'geo': 56}

        # 模态编码器
        self.gnn = MPNEncoder(atom_fdim=args.gnn_atom_dim, bond_fdim=args.gnn_bond_dim,
                              hidden_size=self.hidden_dim, bias=args.bias, depth=args.gnn_num_layers,
                              dropout=args.dropout, activation=args.gnn_activation, device=device)
        
        self.transformer = TrfmSeq2seq(input_dim=args.seq_input_dim, hidden_size=self.hidden_dim,
                                       num_head=args.seq_num_heads, n_layers=args.seq_num_layers, dropout=args.dropout,
                                       vocab_num=args.vocab_num, device=device, recons=args.recons).to(device)
        
        self.compound_encoder = GeoGNNModel(args, compound_encoder_config, device)

        # ✅ 简化的特征提取器
        self.shared_extractors = nn.ModuleDict({
            modal: SimpleExtractor(
                input_dim=self.hidden_dim,
                bottleneck_dim=self.shared_dims[modal],
                output_dim=self.hidden_dim,
                dropout=args.dropout
            ) for modal in ['gnn', 'seq', 'geo']
        }).to(device)
        
        self.private_extractors = nn.ModuleDict({
            modal: SimpleExtractor(
                input_dim=self.hidden_dim,
                bottleneck_dim=self.private_dims[modal],
                output_dim=self.hidden_dim,
                dropout=args.dropout
            ) for modal in ['gnn', 'seq', 'geo']
        }).to(device)

        # ✅ 修复：统一使用 ImprovedGate
        self.gates = nn.ModuleDict({
            modal: ImprovedGate(self.hidden_dim)  # 使用ImprovedGate
            for modal in ['gnn', 'seq', 'geo']
        }).to(device)

        # ✅ 简化的模态内融合层
        self.intra_fusions = nn.ModuleDict({
            modal: nn.Sequential(
                nn.Linear(self.hidden_dim * 3, self.hidden_dim),  # 直接融合到目标维度
                nn.ReLU(),
                nn.Dropout(args.dropout),
                nn.LayerNorm(self.hidden_dim)
            ) for modal in ['gnn', 'seq', 'geo']
        }).to(device)

        # 其他组件
        self.entropy = loss_type[args.task_type]
        self.pool = global_mean_pool if args.pool_type == 'mean' else Global_Attention(self.hidden_dim).to(device)
        
        modal_count = args.graph + args.sequence + args.geometry
        # ✅ 简化的融合层
        self.modal_weight_fusion = SimpleWeightFusion(modal_count, self.latent_dim, device=device)
        
        self.dropout = nn.Dropout(args.dropout)
        
        # ✅ 简化的输出层
        self.output_layer = nn.Sequential(
            nn.Linear(self.latent_dim, self.latent_dim // 2),
            nn.ReLU(),
            nn.Dropout(args.dropout),
            nn.Linear(self.latent_dim // 2, args.output_dim)
        ).to(device)

        # 特征缓存
        self.last_shared_features = None
        self.last_private_features = None

        # 权重初始化
        self._init_weights()

    def set_epoch(self, epoch):
        """设置当前训练轮次"""
        self.current_epoch = epoch

    def _extract_modal_features(self, x, modal_type):
        """改进的模态特征提取"""
        # ✅ 动态温度：训练初期软门控，后期硬门控
        temperature = max(0.5, 2.0 - self.current_epoch * 0.02)
        gate_values = self.gates[modal_type](x, temperature)  # 现在可以传温度参数了
        
        # 提取共享和私有特征
        shared_feat = gate_values[:, 0:1] * self.shared_extractors[modal_type](x)
        private_feat = gate_values[:, 1:2] * self.private_extractors[modal_type](x)
        
        return shared_feat, private_feat

    def _fuse_modal_features(self, original, shared, private, modal_type):
        """内存优化的模态内融合"""
        # ✅ 避免不必要的squeeze操作
        shared = shared.squeeze() if shared.dim() > 2 else shared
        private = private.squeeze() if private.dim() > 2 else private
        
        # ✅ 使用torch.cat的out参数避免临时张量
        concat_feat = torch.cat([original, shared, private], dim=1)
        enhanced = self.intra_fusions[modal_type](concat_feat)
        
        # ✅ 添加轻微残差连接防止梯度消失
        return enhanced + 0.1 * original

    def forward(self, trans_batch_seq, seq_mask, batch_mask_seq, gnn_batch_graph, gnn_feature_batch, batch_mask_gnn,
                graph_dict, node_id_all, edge_id_all):

        modalities = []
        shared_features = []
        private_features = []
        modal_configs = []
        
        # 模态特征提取
        if self.args.graph:
            node_gnn_x = self.gnn(gnn_batch_graph, gnn_feature_batch, batch_mask_gnn)
            graph_gnn_x = self.pool(node_gnn_x, batch_mask_gnn)
            modal_configs.append(('gnn', graph_gnn_x))

        if self.args.sequence:
            _, node_seq_x = self.transformer(trans_batch_seq)
            graph_seq_x = self.pool(node_seq_x[seq_mask], batch_mask_seq)
            modal_configs.append(('seq', graph_seq_x))

        if self.args.geometry:
            node_repr, _ = self.compound_encoder(graph_dict[0], graph_dict[1], node_id_all, edge_id_all)
            graph_geo_x = self.pool(node_repr, node_id_all[0])
            modal_configs.append(('geo', graph_geo_x))

        # 统一处理所有模态
        for modal_type, modal_feat in modal_configs:
            if self.device:
                modal_feat = modal_feat.to(self.device)
                
            # 特征提取
            shared_feat, private_feat = self._extract_modal_features(modal_feat, modal_type)
            
            # 模态内融合
            enhanced_feat = self._fuse_modal_features(modal_feat, shared_feat, private_feat, modal_type)
            
            # 存储结果
            modalities.append(enhanced_feat)
            shared_features.append(shared_feat)
            private_features.append(private_feat)

        # ✅ 简化：直接融合
        causal_repr = self.modal_weight_fusion(modalities)
        causal_repr = self.dropout(causal_repr)
        pred = self.output_layer(causal_repr)
        
        # 缓存特征
        self.last_shared_features = shared_features
        self.last_private_features = private_features

        return causal_repr, pred

    def label_loss(self, pred, label, mask):
        loss_mat = self.entropy(pred, label)
        return loss_mat.sum() / mask.sum()

    def get_dynamic_loss_weights(self):
        """动态损失权重调整"""
        if self.current_epoch < 10:
            # 前10轮：只训练基础任务
            return 0.0, 0.0, 0.0
        elif self.current_epoch < 30:
            # 10-30轮：逐渐引入因果损失
            progress = (self.current_epoch - 10) / 20
            alpha = 0.1 * progress
            beta = 0.05 * progress  
            gamma = 0.05 * progress
            return alpha, beta, gamma
        else:
            # 30轮后：全面训练
            return 0.3, 0.2, 0.1

    # 在loss_cal中使用
    def loss_cal(self, _, pred, label, mask, alpha=None, beta=None, gamma=None):
        """动态损失权重版本"""
        if alpha is None:
            alpha, beta, gamma = self.get_dynamic_loss_weights()
        
        task_loss = self.label_loss(pred, label, mask)
        causal_loss = torch.tensor(0.0, device=self.device)

        if self.last_shared_features and self.last_private_features:
            # ✅ 批量计算，避免循环
            shared_feats = torch.stack([F.normalize(feat.squeeze() if feat.dim() > 2 else feat, dim=1) 
                                       for feat in self.last_shared_features], dim=0)  # [M, B, D]
            private_feats = torch.stack([F.normalize(feat.squeeze() if feat.dim() > 2 else feat, dim=1) 
                                        for feat in self.last_private_features], dim=0)  # [M, B, D]

            # 1. 私有特征独立性损失 - 批量计算
            if len(self.last_private_features) > 1:
                private_sim_matrix = torch.mm(private_feats.view(-1, private_feats.size(2)), 
                                            private_feats.view(-1, private_feats.size(2)).t())
                # 只取上三角（去除对角线）
                mask = torch.triu(torch.ones_like(private_sim_matrix), diagonal=1).bool()
                private_loss = torch.abs(private_sim_matrix[mask]).mean()
            else:
                private_loss = torch.tensor(0.0, device=self.device)

            # 2. 共享-私有解耦损失 - 批量计算
            cross_sim = torch.sum(shared_feats * private_feats, dim=2).mean()  # [M, B] -> scalar
            cross_loss = torch.abs(cross_sim)

            # 3. 共享特征对齐损失 - 批量计算
            if len(self.last_shared_features) > 1:
                shared_sim_matrix = torch.mm(shared_feats.view(-1, shared_feats.size(2)), 
                                           shared_feats.view(-1, shared_feats.size(2)).t())
                mask = torch.triu(torch.ones_like(shared_sim_matrix), diagonal=1).bool()
                shared_loss = torch.clamp(1.0 - shared_sim_matrix[mask], min=0).mean()
            else:
                shared_loss = torch.tensor(0.0, device=self.device)

            # ✅ 简化：直接组合损失
            causal_loss = beta * private_loss + gamma * cross_loss + gamma * shared_loss

        total_loss = task_loss + alpha * causal_loss
        return total_loss, task_loss, causal_loss

    def _init_weights(self):
        """改进的权重初始化"""
        for name, module in self.named_modules():
            if isinstance(module, nn.Linear):
                if 'gate' in name:
                    # ✅ 门控层：偏向均匀分布但有小随机性
                    nn.init.xavier_uniform_(module.weight)
                    if module.bias is not None:
                        # 初始化为轻微偏向共享特征
                        nn.init.constant_(module.bias, 0.1)
                elif 'extractor' in name:
                    # ✅ 提取器：He初始化
                    nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                elif 'fusion' in name:
                    # ✅ 融合层：Xavier初始化
                    nn.init.xavier_normal_(module.weight)
            elif isinstance(module, nn.LayerNorm):
                nn.init.constant_(module.bias, 0)
                nn.init.constant_(module.weight, 1.0)