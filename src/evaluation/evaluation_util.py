import numpy as np
import torch

def evaluate_family_classifier(model, test_loader, loss_function=None, batch_size=None,
                                 mode='test', device='cpu', verbose=False):
    model.eval()
    with torch.no_grad():
        losses = []
        accuracies = []

        for batch_idx, data in enumerate(test_loader):
            data.x = data.x.to(device)
            data.edge_index = data.edge_index.to(device)
            data.edge_attr = data.edge_attr.to(device)
            data.batch = data.batch.to(device)
            data.y = data.y.to(device)

            out = model(data)

            # Loss is computed with respect to the target sequence
            loss = loss_function(out, data.y)
            losses.append(loss.item())

            pred = out.max(1)[1]
            accuracy = compute_metrics_family(data.y, pred)

            accuracies.append(accuracy)

        avg_loss = np.mean(losses)
        avg_accuracy = np.mean(accuracies)

        print("{} loss: {}".format(mode, avg_loss))
        print("{} accuracy: {}".format(mode, avg_accuracy))

        return avg_loss, avg_accuracy


def compute_metrics_family(target, pred):
    # accuracy = accuracy_score(target, pred)
    accuracy = (target.eq(pred.long())).sum().item() / target.shape[0]

    return accuracy


def get_sensitivity(cf):
    # tp / (tp + fn)
    sensitivity = 0
    for c in range(len(cf)):
        tp = cf[c, c]
        fn = np.sum(cf[:, c]) - tp
        sensitivity += tp / (tp + fn)

    return sensitivity / len(cf)


def get_specificity(cf):
    specificity = 0
    for c in range(len(cf)):
        tp = cf[c, c]
        fp = np.sum(cf[c, :]) - tp
        fn = np.sum(cf[:, c]) - tp
        tn = np.sum(cf) - tp - fp - fn
        specificity += tn / (tn + fp)

    return specificity / len(cf)
