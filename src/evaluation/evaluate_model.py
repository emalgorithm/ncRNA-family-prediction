import os
import sys
sys.path.append(os.getcwd().split('src')[0])

from sklearn.metrics import classification_report, matthews_corrcoef, accuracy_score, confusion_matrix
import pickle
import torch
import argparse

from src.data_util.data_constants import families, word_to_ix
from src.data_util.rna_family_graph_dataset import RNAFamilyGraphDataset
from torch_geometric.data import DataLoader
from src.model.gcn import GCN
from src.evaluation.evaluation_util import get_sensitivity, get_specificity

parser = argparse.ArgumentParser()
parser.add_argument('--model_name', default="test", help='model name')
parser.add_argument('--test_dataset',
                    default='../data/test_13_classes.fasta', help='Path to test dataset')
args = parser.parse_args()

foldings_dataset = '../data/foldings.pkl'

device = 'cuda' if torch.cuda.is_available() else 'cpu'
n_classes = len(families)

test_set = RNAFamilyGraphDataset(args.test_dataset, foldings_dataset)
test_loader = DataLoader(test_set, batch_size=64, shuffle=False)

opt = pickle.load(open('../results_family_classification/' + args.model_name +
                       '/hyperparams.pkl', "rb"))

model = GCN(n_features=opt.embedding_dim, hidden_dim=opt.hidden_dim, n_classes=n_classes,
            n_conv_layers=opt.n_conv_layers,
            dropout=opt.dropout, batch_norm=opt.batch_norm, num_embeddings=len(word_to_ix),
            embedding_dim=opt.embedding_dim,
            node_classification=False,
            set2set_pooling=opt.set2set_pooling).to(opt.device)

model.load_state_dict(torch.load('../models_family_classification/' + args.model_name + '/model.pt',
                                 map_location=device))
print("The model has {} parameters".format(sum(p.numel() for p in model.parameters() if p.requires_grad)))

y_pred = []
y_true = []

for batch_idx, data in enumerate(test_loader):
    model.eval()

    data.x = data.x.to(device)
    data.edge_index = data.edge_index.to(device)
    data.edge_attr = data.edge_attr.to(device)
    data.batch = data.batch.to(device)
    data.y = data.y.to(device)

    out = model(data)

    pred = out.max(1)[1]

    y_pred += list(pred.cpu().numpy())
    y_true += list(data.y.cpu().numpy())

print(classification_report(y_true, y_pred, target_names=families, digits=4))
print("Accuracy: {0:.4f}".format(accuracy_score(y_true, y_pred)))
print("Sensitivity: {0:.4f}".format(get_sensitivity(confusion_matrix(y_true, y_pred))))
print("Specificity: {0:.4f}".format(get_specificity(confusion_matrix(y_true, y_pred))))
print("MCC: {0:.4f}".format(matthews_corrcoef(y_true, y_pred)))
