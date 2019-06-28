import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, NNConv, GINConv, GATConv, global_add_pool, Set2Set


class GCN(nn.Module):
    def __init__(self, n_features, hidden_dim, n_classes, n_conv_layers=3, dropout=0,
                 conv_type="MPNN", set2set_pooling=False, node_classification=True, softmax=False,
                 probability=True, batch_norm=True, num_embeddings=None, embedding_dim=20,
                 residuals=False, device="cpu"):
        super(GCN, self).__init__()
        if num_embeddings:
            self.embedding = nn.Embedding(num_embeddings=num_embeddings, embedding_dim=embedding_dim)

        self.convs = nn.ModuleList()
        self.batch_norms = nn.ModuleList()

        # First layer
        self.convs.append(self.get_conv_layer(n_features, hidden_dim, conv_type=conv_type))
        self.batch_norms.append(nn.BatchNorm1d(hidden_dim))

        # Hidden layers
        for i in range(n_conv_layers - 1):
            self.convs.append(self.get_conv_layer(hidden_dim, hidden_dim, conv_type=conv_type))
            self.batch_norms.append(nn.BatchNorm1d(hidden_dim))

        # Fully connected layer
        self.fc = nn.Linear(hidden_dim, n_classes)

        # if residuals:
        #     self.fc = nn.Linear(n_conv_layers * hidden_dim, n_classes)

        # If we are interested in graph classification, we introduce the final pooling and change
        # the fc layer to have dimensions compatible with the output of the Set2Set model
        if set2set_pooling:
            self.fc = nn.Linear(2 * hidden_dim, n_classes)
            self.pooling = Set2Set(hidden_dim, processing_steps=10)

        self.dropout = nn.Dropout(dropout)

        self.conv_type = conv_type
        self.node_classification = node_classification
        self.softmax = softmax
        self.probability = probability
        self.batch_norm = batch_norm
        self.num_embeddings = num_embeddings
        self.set2set_pooling = set2set_pooling
        self.residuals = residuals
        self.device = device

    def forward(self, data):
        x, adj, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
        # poolings = torch.Tensor([]).to(self.device)

        if self.num_embeddings:
            x = self.embedding(x)
            x = self.dropout(x)

        # Apply graph convolutional layers
        for i, conv in enumerate(self.convs):
            x = self.apply_conv_layer(conv, x, adj, edge_attr, conv_type=self.conv_type)
            x = self.batch_norms[i](x) if self.batch_norm else x
            x = nn.functional.leaky_relu(x)
            x = self.dropout(x)

            # if self.residuals:
            #     poolings = torch.cat((poolings, global_add_pool(x, batch)), dim=1)

        # If we are interested in graph classification, apply graph-wise pooling
        if not self.node_classification: # and not self.residuals:
            if self.set2set_pooling:
                x = self.pooling(x, batch)
            else:
                x = global_add_pool(x, batch)
            x = self.dropout(x)

        # if self.residuals:
        #     x = self.dropout(poolings)

        x = self.fc(x)

        # if not self.node_classification:
        #     if self.probability:
        #         return torch.sigmoid(x)
        #     else:
        #         return x

        return F.log_softmax(x, dim=1) if not self.softmax else F.softmax(x, dim=1)

    @staticmethod
    def get_conv_layer(n_input_features, n_output_features, conv_type="GCN"):
        if conv_type == "GCN":
            return GCNConv(n_input_features, n_output_features)
        elif conv_type == "GAT":
            return GATConv(n_input_features, n_output_features)
        elif conv_type == "MPNN":
            net = nn.Sequential(nn.Linear(2, 10), nn.ReLU(), nn.Linear(10, n_input_features *
                                                                      n_output_features))
            return NNConv(n_input_features, n_output_features, net)
        elif conv_type == "GIN":
            net = nn.Sequential(nn.Linear(n_input_features, n_output_features), nn.ReLU(),
                                nn.Linear(n_output_features, n_output_features))
            return GINConv(net)
        else:
            raise Exception("{} convolutional layer is not supported.".format(conv_type))

    @staticmethod
    def apply_conv_layer(conv, x, adj, edge_attr, conv_type="GCN"):
        if conv_type in ["GCN", "GAT", "GIN"]:
            return conv(x, adj)
        elif conv_type in ["MPNN"]:
            return conv(x, adj, edge_attr)
        else:
            raise Exception("{} convolutional layer is not supported.".format(conv_type))
