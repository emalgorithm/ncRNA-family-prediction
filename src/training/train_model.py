import os
import sys
sys.path.append(os.getcwd().split('src')[0])

import torch
import torch.nn as nn
import torch.optim as optim
import pickle
import os
import time
import argparse
import numpy as np
from torch_geometric.data import DataLoader

from src.model.gcn import GCN
from src.data_util.rna_family_graph_dataset import RNAFamilyGraphDataset
from src.util.visualization_util import plot_loss
from src.data_util.data_constants import word_to_ix, families
from src.evaluation.evaluation_util import evaluate_family_classifier, compute_metrics_family

torch.manual_seed(0)
np.random.seed(0)

parser = argparse.ArgumentParser()
parser.add_argument('--model_name', default="test", help='model name')
parser.add_argument('--device', default="cpu", help='cpu or cuda')
parser.add_argument('--n_samples', type=int, default=None, help='Number of samples to train on')
parser.add_argument('--n_epochs', type=int, default=10000, help='Number of samples to train on')
parser.add_argument('--embedding_dim', type=int, default=20, help='Dimension of nucleotide '
                                                                  'embeddings')
parser.add_argument('--hidden_dim', type=int, default=80, help='Dimension of hidden '
                                                                'representations of convolutional layers')
parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
parser.add_argument('--learning_rate', type=float, default=0.0004, help='Learning rate')
parser.add_argument('--seq_max_len', type=int, default=10000, help='Maximum length of sequences '
                                                                 'used for training and testing')
parser.add_argument('--seq_min_len', type=int, default=1, help='Maximum length of sequences '
                                                                 'used for training and testing')
parser.add_argument('--n_conv_layers', type=int, default=5, help='Number of convolutional layers')
parser.add_argument('--conv_type', type=str, default="MPNN", help='Type of convolutional layers')
parser.add_argument('--dropout', type=float, default=0.1, help='Amount of dropout')
parser.add_argument('--batch_norm', dest='batch_norm', action='store_true')
parser.add_argument('--no_batch_norm', dest='batch_norm', action='store_false')
parser.set_defaults(batch_norm=True)
parser.add_argument('--residuals', type=bool, default=False, help='Whether to use residuals')
parser.add_argument('--set2set_pooling', type=bool, default=True, help='Whether to use set2set '
                                                                        'pooling')
parser.add_argument('--early_stopping', type=int, default=30, help='Number of epochs for early '
                                                                   'stopping')
parser.add_argument('--verbose', type=bool, default=False, help='Verbosity')
parser.add_argument('--foldings_dataset', type=str,
                    default='../data/foldings.pkl', help='Path to foldings')
parser.add_argument('--train_dataset', type=str,
                    default='../data/train.fasta', help='Path to training '
                                                                          'dataset')
parser.add_argument('--val_dataset', type=str,
                    default='../data/val.fasta', help='Path to val dataset')
parser.add_argument('--test_dataset', type=str,
                    default='../data/test_13_classes.fasta', help='Path to test dataset')


opt = parser.parse_args()
print(opt)

n_classes = len(families)
model = GCN(n_features=opt.embedding_dim, hidden_dim=opt.hidden_dim, n_classes=n_classes,
            n_conv_layers=opt.n_conv_layers,
            dropout=opt.dropout, batch_norm=opt.batch_norm, num_embeddings=len(word_to_ix),
            embedding_dim=opt.embedding_dim,
            node_classification=False, residuals=opt.residuals, device=opt.device,
            set2set_pooling=opt.set2set_pooling).to(opt.device)

loss_function = nn.NLLLoss()
optimizer = optim.Adam(model.parameters(), lr=opt.learning_rate)

# Data Loading
n_train_samples = None if not opt.n_samples else int(opt.n_samples * 0.8)
n_val_samples = None if not opt.n_samples else int(opt.n_samples * 0.1)
n_test_samples = None if not opt.n_samples else int(opt.n_samples * 0.1)

train_set = RNAFamilyGraphDataset(opt.train_dataset, opt.foldings_dataset,
                                  seq_max_len=opt.seq_max_len,
                                    seq_min_len=opt.seq_min_len,
                                    n_samples=n_train_samples)
val_set = RNAFamilyGraphDataset(opt.val_dataset, opt.foldings_dataset, seq_max_len=opt.seq_max_len,
                                seq_min_len=opt.seq_min_len,
                                n_samples=n_val_samples)
test_set = RNAFamilyGraphDataset(opt.test_dataset, opt.foldings_dataset,
                                 seq_max_len=opt.seq_max_len,
                                 seq_min_len=opt.seq_min_len,
                                 n_samples=n_test_samples)

train_loader = DataLoader(train_set, batch_size=opt.batch_size, shuffle=True)
val_loader = DataLoader(val_set, batch_size=opt.batch_size, shuffle=False)
test_loader = DataLoader(test_set, batch_size=opt.batch_size, shuffle=False)


def train_epoch(model, train_loader):
    model.train()
    losses = []
    accuracies = []

    for batch_idx, data in enumerate(train_loader):
        data.x = data.x.to(opt.device)
        data.edge_index = data.edge_index.to(opt.device)
        data.edge_attr = data.edge_attr.to(opt.device)
        data.batch = data.batch.to(opt.device)
        data.y = data.y.to(opt.device)

        model.zero_grad()

        out = model(data)

        # Loss is computed with respect to the target sequence
        loss = loss_function(out, data.y)
        losses.append(loss.item())
        loss.backward()
        optimizer.step()

        # Metrics are computed with respect to generated folding
        pred = out.max(1)[1]
        accuracy = compute_metrics_family(data.y, pred)
        accuracies.append(accuracy)

    avg_loss = np.mean(losses)
    avg_accuracy = np.mean(accuracies)

    print("training loss is {}".format(avg_loss))
    print("accuracy: {}".format(avg_accuracy))

    return avg_loss.item(), avg_accuracy


def run(model, n_epochs, train_loader, results_dir, model_dir):
    print("The model contains {} parameters".format(sum(p.numel() for p in model.parameters() if p.requires_grad)))

    train_losses = []
    train_accuracies = []
    val_losses = []
    val_accuracies = []
    test_losses = []
    test_accuracies = []

    for epoch in range(n_epochs):
        start = time.time()
        print("Epoch {}: ".format(epoch + 1))

        loss, accuracy = train_epoch(model, train_loader)
        val_loss, val_accuracy = evaluate_family_classifier(model, val_loader,
                                                                          loss_function, mode='val',
                                                                    device=opt.device, verbose=opt.verbose)
        test_loss, test_accuracy = evaluate_family_classifier(model, test_loader,
                                                              loss_function, mode='test',
                                                              device=opt.device,
                                                              verbose=opt.verbose)
        end = time.time()
        print("Epoch took {0:.2f} seconds".format(end - start))

        if not val_accuracies or val_accuracy > max(val_accuracies):
            torch.save(model.state_dict(), model_dir + 'model.pt')
            print("Saved updated model")
        #
        train_losses.append(loss)
        val_losses.append(val_loss)
        test_losses.append(test_loss)
        train_accuracies.append(accuracy)
        val_accuracies.append(val_accuracy)
        test_accuracies.append(test_accuracy)

        plot_loss(train_losses, val_losses, test_losses ,file_name=results_dir + 'loss.jpg')
        plot_loss(train_accuracies, val_accuracies, test_accuracies ,file_name=results_dir + 'acc.jpg',
                  y_label='accuracy')

        pickle.dump({
            'train_losses': train_losses,
            'val_losses': val_losses,
            'test_losses': test_losses,
            'train_accuracies': train_accuracies,
            'val_accuracies': val_accuracies,
            'test_accuracies': test_accuracies,
        }, open(results_dir + 'scores.pkl', 'wb'))

        if len(val_accuracies) > opt.early_stopping and max(val_accuracies[-opt.early_stopping:])\
                < max(val_accuracies):
            print("Training terminated because of early stopping")
            print("Best val_loss: {}".format(min(val_losses)))
            print("Best val_accuracy: {}".format(max(val_accuracies)))
            print("Corresponding test_loss: {}".format(test_losses[np.argmin(val_losses)]))
            print("Corresponding test_accuracy: {}".format(test_accuracies[np.argmax(
                val_accuracies)]))

            with open(results_dir + 'scores.txt', 'w') as f:
                f.write("Best val_accuracy: {} \n Corresponding test_accuracy: {}".format(max(
                    val_accuracies), test_accuracies[np.argmax(
                val_accuracies)]))
            break


def main():
    results_dir = '../results_family_classification/{}/'.format(opt.model_name)
    model_dir = '../models_family_classification/{}/'.format(opt.model_name)
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)

    with open(results_dir + 'hyperparams.txt', 'w') as f:
        f.write(str(opt))

    with open(results_dir + 'hyperparams.pkl', 'wb') as f:
        pickle.dump(opt, f)

    run(model, opt.n_epochs, train_loader, results_dir, model_dir)


if __name__ == "__main__":
    main()

