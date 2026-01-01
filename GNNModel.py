# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, GCNConv
from logger import debug, debug_print, set_debug_mode
from Parameters import *
import Parameters


class EnhancedHeteroGNN(nn.Module):
    def __init__(self, node_feature_dim=9, hidden_dim=64, num_heads=4, num_layers=2, dropout=0.2):
        super(EnhancedHeteroGNN, self).__init__()

        # 动态读取当前架构模式
        self.arch_type = getattr(Parameters, 'GNN_ARCH', 'HYBRID')
        debug_print(f"Initializing GNN Model with Architecture: {self.arch_type}")

        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout

        from GraphBuilder import global_graph_builder
        self.edge_feature_dim = global_graph_builder.comm_edge_feature_dim
        self.edge_types = ['communication', 'interference', 'proximity']

        # 1. 节点嵌入
        self.node_type_embedding = nn.Embedding(2, hidden_dim // 4)

        # 2. 定义图卷积层 (根据架构不同)
        self.edge_type_layers = nn.ModuleDict()

        for edge_type in self.edge_types:
            layers = nn.ModuleList()
            input_dim = node_feature_dim + (hidden_dim // 4)

            for i in range(num_layers):
                curr_in = input_dim if i == 0 else hidden_dim
                curr_out = hidden_dim // num_heads if (self.arch_type != "GCN" and i < num_layers - 1) else hidden_dim

                if self.arch_type == "GCN":
                    # GCN 不支持多头，且处理边特征较弱
                    layers.append(GCNConv(curr_in, hidden_dim))
                else:
                    # GAT 和 HYBRID 使用 GATConv
                    heads = num_heads if i < num_layers - 1 else 1
                    concat = True if i < num_layers - 1 else False
                    layers.append(GATConv(curr_in, curr_out, heads=heads, dropout=dropout,
                                          edge_dim=self.edge_feature_dim, concat=concat))

            self.edge_type_layers[edge_type] = layers

        # 3. HYBRID 专属组件: 边门控 (Edge Gating)
        if self.arch_type == "HYBRID":
            self.edge_type_gates = nn.Parameter(torch.zeros(len(self.edge_types)))

        # 4. 边类型融合权重
        self.edge_type_attention = nn.Parameter(torch.ones(len(self.edge_types)))

        # 5. 输出层
        self.attn_pool_linear = nn.Linear(hidden_dim, 1)
        self.output_layer = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, RL_N_ACTIONS)
        )

    def forward(self, graph_data, dqn_id=None):
        node_features = graph_data['node_features']['features']
        node_types = graph_data['node_features']['types']
        edge_features = graph_data['edge_features']

        batch_size = node_features.size(0)
        type_embedding = self.node_type_embedding(node_types)
        x = torch.cat([node_features, type_embedding], dim=1)

        edge_outputs = []

        # HYBRID 模式计算 Gate
        edge_gates = torch.sigmoid(self.edge_type_gates) if self.arch_type == "HYBRID" else None

        # 边权重 (HYBRID 和 GAT 都用，GCN 用平均)
        edge_weights = F.softmax(self.edge_type_attention, dim=0)

        for i, edge_type in enumerate(self.edge_types):
            if edge_features[edge_type] is None:
                edge_outputs.append(torch.zeros(batch_size, self.hidden_dim, device=x.device))
                continue

            edge_index = edge_features[edge_type]['edge_index']
            edge_attr = edge_features[edge_type]['edge_attr']

            # --- 架构分支逻辑 ---
            if self.arch_type == "HYBRID":
                # 只有 Hybrid 使用 Gate 对边特征进行缩放
                gated_edge_attr = edge_attr * edge_gates[i]
            else:
                # GAT 和 GCN 直接使用原始边特征
                gated_edge_attr = edge_attr

            x_edge = x.clone()
            layers = self.edge_type_layers[edge_type]

            for j, layer in enumerate(layers):
                if self.arch_type == "GCN":
                    # GCNConv 无法直接处理多维 edge_attr，这里我们做一个简化：
                    # 只利用图结构信息 (Topology)，忽略具体的 CSI 数值
                    # 这也是 GCN 在通信场景通常弱于 GAT 的原因
                    x_edge = layer(x_edge, edge_index)
                else:
                    # GAT / HYBRID
                    x_edge = layer(x_edge, edge_index, edge_attr=gated_edge_attr)

                if j < len(layers) - 1:
                    x_edge = F.elu(x_edge)
                    x_edge = F.dropout(x_edge, p=self.dropout, training=self.training)

            # 聚合不同边类型
            if self.arch_type == "GCN":
                # GCN 简单相加
                edge_outputs.append(x_edge)
            else:
                # GAT/HYBRID 使用可学习权重
                edge_outputs.append(x_edge * edge_weights[i])

        # 聚合
        if len(edge_outputs) > 0:
            stacked = torch.stack(edge_outputs, dim=0)
            x_combined = torch.sum(stacked, dim=0) if self.arch_type == "GCN" else torch.sum(stacked, dim=0)
        else:
            x_combined = torch.zeros(batch_size, self.hidden_dim, device=x.device)

        # 输出 Q 值
        if dqn_id is not None:
            q_values = self._extract_local_features(x_combined, graph_data, dqn_id)
        else:
            q_values = self._extract_global_features(x_combined, graph_data)

        # 返回 (Q值, 辅助信息)
        # 只有 Hybrid 模式返回 attention logits 用于计算 Entropy Loss
        # 其他模式返回 None，Main.py 需处理
        aux_info = self.edge_type_attention if self.arch_type == "HYBRID" else None

        return q_values, aux_info

    def _extract_local_features(self, node_embeddings, graph_data, dqn_id):
        nodes = graph_data['nodes']
        target_rsu_index = -1
        for i, rsu_node in enumerate(nodes['rsu_nodes']):
            if rsu_node['original_id'] == dqn_id:
                target_rsu_index = i
                break
        if target_rsu_index == -1:
            return torch.zeros(RL_N_ACTIONS, device=node_embeddings.device)

        rsu_embedding = node_embeddings[target_rsu_index]
        vehicle_embeddings = []

        for vehicle_node in nodes['vehicle_nodes']:
            for edge in graph_data['edges']['communication']:
                if (edge['source'] == f"rsu_{dqn_id}" and edge['target'] == vehicle_node['id']):
                    vehicle_index = len(nodes['rsu_nodes']) + nodes['vehicle_nodes'].index(vehicle_node)
                    vehicle_embeddings.append(node_embeddings[vehicle_index])
                    break

        if vehicle_embeddings:
            vehicle_stack = torch.stack(vehicle_embeddings)
            attn_scores = self.attn_pool_linear(vehicle_stack)
            attn_weights = F.softmax(attn_scores, dim=0)
            vehicle_embedding = torch.mm(attn_weights.t(), vehicle_stack).squeeze(0)
        else:
            vehicle_embedding = torch.zeros_like(rsu_embedding)

        combined_features = torch.cat([rsu_embedding, vehicle_embedding], dim=0)
        q_values = self.output_layer(combined_features)
        return q_values

    def _extract_global_features(self, node_embeddings, graph_data):
        nodes = graph_data['nodes']
        num_rsus = len(nodes['rsu_nodes'])
        all_q_values = []
        for dqn_id in range(1, num_rsus + 1):
            q_value = self._extract_local_features(node_embeddings, graph_data, dqn_id)
            all_q_values.append(q_value)
        if all_q_values:
            return torch.stack(all_q_values, dim=0)
        else:
            return torch.zeros(0, RL_N_ACTIONS, device=node_embeddings.device)

    def get_attention_weights(self, graph_data):
        # GCN 没有内部注意力，只返回边类型权重
        attention_info = {
            'edge_type_weights': F.softmax(self.edge_type_attention, dim=0).detach().cpu().numpy(),
            'edge_types': self.edge_types
        }
        return attention_info


# 显式传入参数，确保 GAT/Hybrid 模式下多头注意力生效
global_gnn_model = EnhancedHeteroGNN(
    node_feature_dim=12,
    hidden_dim=64,
    num_heads=4,
    num_layers=2,
    dropout=0.2
)

global_target_gnn_model = EnhancedHeteroGNN(
    node_feature_dim=12,
    hidden_dim=64,
    num_heads=4,
    num_layers=2,
    dropout=0.2
)

def update_target_gnn():
    global_target_gnn_model.load_state_dict(global_gnn_model.state_dict())
    global_target_gnn_model.eval()
    debug(f"Global Target GNN ({getattr(Parameters, 'GNN_ARCH', 'Hybrid')}) updated")

def update_target_gnn_soft(tau):
    try:
        with torch.no_grad():
            for target_param, online_param in zip(global_target_gnn_model.parameters(), global_gnn_model.parameters()):
                target_param.data.copy_(tau * online_param.data + (1.0 - tau) * target_param.data)
    except Exception as e:
        debug(f"Error during GNN soft update: {e}")

# 初始化并同步
update_target_gnn()
debug_print(f"Global GNN ({getattr(Parameters, 'GNN_ARCH', 'Hybrid')}) initialized and synced.")

if __name__ == "__main__":
    set_debug_mode(True)
    debug_print("GNNModel.py (GCN Version) loaded.")