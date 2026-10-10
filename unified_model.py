import torch
import torch.nn as nn
import torch_geometric.nn as hnn
import dgl
from dgl.nn import DGNConv

class FeatureMappingMLP(nn.Module):

    def __init__(self, in_dim=2304, out_dim=512):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, out_dim)
        )
    
    def forward(self, x):
        return self.mlp(x)

class MetabolicDirectedGraphLearning(nn.Module):
    
    def __init__(self, in_channels=512, hid_channels=256, num_dgnn=2):
        super().__init__()
        self.num_dgnn = num_dgnn
        self.layers = nn.ModuleList()
        
        if self.num_dgnn == 1:
            self.layers.append(
                DGNConv(in_channels, hid_channels, ['dir2-av', 'dir2-dx', 'sum'], 
                        ['identity', 'amplification'], 2.5))
        else:
            self.layers.append(
                DGNConv(in_channels, hid_channels, ['dir3-av', 'dir3-dx', 'sum'], 
                        ['identity', 'amplification'], 2.5))
            for i in range(self.num_dgnn - 1):
                self.layers.append(
                    DGNConv(hid_channels, hid_channels, ['dir3-av', 'dir3-dx', 'sum'], 
                            ['identity', 'amplification'], 2.5))
    
    def forward(self, g, features, eig_vec):
        x = features
        for i in range(self.num_dgnn):
            x = self.layers[i](g, x, eig_vec=eig_vec)
        return x

class UnifiedDuoMet(nn.Module):
    
    def __init__(self, 
                 biochemical_in_dim=2304,
                 biochemical_out_dim=512,
                 dgnn_hid_dim=256,
                 num_dgnn=2,
                 
                 input_num=None,
                 hyper_in_dim=None,
                 conv_dim=128,
                 head=6,
                 p=0.1,
                 L=2,
                 use_attention=True,
                 similarity=None):
        
        super(UnifiedDuoMet, self).__init__()
        
        self.biochemical_module = FeatureMappingMLP(biochemical_in_dim, biochemical_out_dim)
        self.directed_graph_module = MetabolicDirectedGraphLearning(
            biochemical_out_dim, dgnn_hid_dim, num_dgnn)
        
        self.emb_dim = dgnn_hid_dim
        self.conv_dim = conv_dim
        self.p = p
        self.input_num = input_num
        self.head = head
        self.hyper_conv_L = L
        self.similarity = similarity
        
        self.in_channel = self.emb_dim
        if similarity is not None:
            self.in_channel = 2 * self.emb_dim
            self.similarity_liner = nn.Linear(input_num, self.emb_dim)
        
        self.relu = nn.ReLU()
        self.hypergraph_conv = hnn.HypergraphConv(
            self.in_channel, conv_dim, heads=head, use_attention=use_attention, dropout=p)
        
        if L > 1:
            self.hypergraph_conv_list = nn.ModuleList()
            for l in range(L - 1):
                self.hypergraph_conv_list.append(
                    hnn.HypergraphConv(head * conv_dim, conv_dim, heads=head, 
                                      use_attention=use_attention, dropout=p))
        
        if use_attention:
            self.hyper_attr_liner = nn.Linear(input_num, self.in_channel)
            if L > 1:
                self.hyperedge_attr_list = nn.ModuleList()
                for l in range(L - 1):
                    self.hyperedge_attr_list.append(nn.Linear(input_num, head * conv_dim))
        
        self.hyperedge_linear = nn.Linear(conv_dim * head, 2)
        self.softmax = nn.Softmax(dim=1)
    
    def forward(self, biochemical_features, g, eig_vec, incidence_matrix):
        biochemical_emb = self.biochemical_module(biochemical_features)
        node_features = self.directed_graph_module(g, biochemical_emb, eig_vec)
        
        if self.similarity is not None:
            simi_feature = self.relu(self.similarity_liner(self.similarity))
            node_features = torch.cat((simi_feature, node_features), dim=1)
        
        row, col = torch.where(incidence_matrix.T)
        edges = torch.cat((col.view(1, -1), row.view(1, -1)), dim=0)
        
        hyperedge_attr = self.hyper_attr_liner(incidence_matrix.T)
        node_features = self.hypergraph_conv(node_features, edges, hyperedge_attr=hyperedge_attr)
        
        if self.hyper_conv_L > 1:
            for l in range(self.hyper_conv_L - 1):
                layer_hyperedge_attr = self.hyperedge_attr_list[l](incidence_matrix.T)
                node_features = self.hypergraph_conv_list[l](node_features, edges,
                                                            hyperedge_attr=layer_hyperedge_attr)
                node_features = self.relu(node_features)
        
        hyperedge_feature = torch.mm(incidence_matrix.T, node_features)
        output = self.hyperedge_linear(hyperedge_feature)
        
        return output
    
    def predict(self, biochemical_features, g, eig_vec, incidence_matrix):
        return self.softmax(self.forward(biochemical_features, g, eig_vec, incidence_matrix))

    def get_node_features(self, biochemical_features, g, eig_vec, incidence_matrix):
        biochemical_emb = self.biochemical_module(biochemical_features)
        node_features = self.directed_graph_module(g, biochemical_emb, eig_vec)
        
        if self.similarity is not None:
            simi_feature = self.relu(self.similarity_liner(self.similarity))
            node_features = torch.cat((simi_feature, node_features), dim=1)
        
        row, col = torch.where(incidence_matrix.T)
        edges = torch.cat((col.view(1, -1), row.view(1, -1)), dim=0)
        hyperedge_attr = self.hyper_attr_liner(incidence_matrix.T)
        node_features = self.hypergraph_conv(node_features, edges, hyperedge_attr=hyperedge_attr)
        
        if self.hyper_conv_L > 1:
            for l in range(self.hyper_conv_L - 1):
                layer_hyperedge_attr = self.hyperedge_attr_list[l](incidence_matrix.T)
                node_features = self.hypergraph_conv_list[l](node_features, edges,
                                                            hyperedge_attr=layer_hyperedge_attr)
                node_features = self.relu(node_features)
        
        return node_features