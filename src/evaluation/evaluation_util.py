from sklearn.metrics import hamming_loss
import numpy as np
import torch
import RNA
from src.data_util.data_processing import one_hot_embed_sequence, prepare_sequences, decode_sequence
from src.data_util.data_constants import word_to_ix, tag_to_ix, ix_to_word, ix_to_tag
from sklearn.metrics import accuracy_score, f1_score, precision_score


def masked_hamming_loss(target, pred, ignore_idx=0):
    mask = target != ignore_idx
    return hamming_loss(target[mask], pred[mask])


def compute_accuracy(target, pred, ignore_idx=0):
    accuracy = 0
    for i in range(len(target)):
        mask = target[i] != ignore_idx
        accuracy += 1 if np.array_equal(target[i][mask], pred[i][mask]) else 0

    return accuracy / len(pred)


def compute_metrics_graph(target_dot_brackets, input_sequences, pred_sequences_scores, batch,
                    verbose=False):
    pred_sequence = pred_sequences_scores.max(1)[1]
    target_dot_brackets_np = []
    pred_sequences_np = []
    input_sequences_np = []

    for j, i in enumerate(batch):
        if len(target_dot_brackets_np) <= i:
            pred_sequences_np.append([])
            target_dot_brackets_np.append([])
            input_sequences_np.append([])
        pred_sequences_np[i].append(pred_sequence[j].item())
        target_dot_brackets_np[i].append(target_dot_brackets[j].item())
        input_sequences_np[i].append(input_sequences[j].item())

    dot_brackets_strings = [decode_sequence(dot_bracket, ix_to_tag) for i, dot_bracket in
                            enumerate(target_dot_brackets_np)]
    sequences_strings = [decode_sequence(sequence, ix_to_word) for i, sequence in enumerate(
        input_sequences_np)]

    pred_sequences_strings = [decode_sequence(pred, ix_to_word).replace('<PAD>', 'A') for i, pred in enumerate(
        pred_sequences_np)]
    pred_dot_brackets_strings = [RNA.fold(pred_sequences_strings[i])[0] for
                                 i, pred_sequence in enumerate(pred_sequences_strings)]

    h_loss = np.mean([hamming_loss(list(dot_brackets_strings[i]),
                                   list(pred_dot_brackets_strings[i])) for i in range(len(
        pred_dot_brackets_strings))])
    accuracy = np.mean([1 if (dot_brackets_strings[i] == pred_dot_brackets_strings[i]) else 0 for i
                       in range(len(pred_dot_brackets_strings))])

    if verbose:
        for i in range(len(dot_brackets_strings)):
            print("REAL SEQUENCE: {}".format(sequences_strings[i]))
            print("PRED SEQUENCE: {}".format(pred_sequences_strings[i]))
            print("REAL: {}".format(dot_brackets_strings[i]))
            print("PRED: {}".format(pred_dot_brackets_strings[i]))
            print()

    return h_loss, accuracy
    # return 0, 0


def compute_metrics(target_dot_brackets, input_sequences, pred_sequences_scores, sequences_lengths,
                    verbose=False):
    dot_brackets_strings = [decode_sequence(dot_bracket.cpu().numpy()[:sequences_lengths[
        i]], ix_to_tag) for i, dot_bracket in enumerate(target_dot_brackets)]
    sequences_strings = [decode_sequence(sequence.cpu().numpy()[:sequences_lengths[
        i]], ix_to_word) for i, sequence in enumerate(input_sequences)]

    pred_sequences_np = pred_sequences_scores.max(2)[1].cpu().numpy()
    pred_sequences_strings = [decode_sequence(pred[:sequences_lengths[i]], ix_to_word) for i,
                                                          pred in enumerate(pred_sequences_np)]
    pred_dot_brackets_strings = [RNA.fold(pred_sequences_strings[i])[0] for
                                 i, pred_sequence in enumerate(pred_sequences_strings)]

    h_loss = np.mean([hamming_loss(list(dot_brackets_strings[i]),
                                   list(pred_dot_brackets_strings[i])) for i in range(len(
        pred_dot_brackets_strings))])
    accuracy = np.mean([1 if (dot_brackets_strings[i] == pred_dot_brackets_strings[i]) else 0 for i
                       in range(len(pred_dot_brackets_strings))])

    if verbose:
        for i in range(len(dot_brackets_strings)):
            print("REAL SEQUENCE: {}".format(sequences_strings[i]))
            print("PRED SEQUENCE: {}".format(pred_sequences_strings[i]))
            print("REAL: {}".format(dot_brackets_strings[i]))
            print("PRED: {}".format(pred_dot_brackets_strings[i]))
            print()

    return h_loss, accuracy
    # return 0, 0


def evaluate(model, test_loader, loss_function, batch_size, mode='test', device='cpu'):
    model.eval()
    with torch.no_grad():
        losses = []
        h_losses = []
        accuracies = []

        for batch_idx, (sequences, dot_brackets, sequences_lengths) in enumerate(test_loader):
            sequences = sequences.to(device)
            dot_brackets = dot_brackets.to(device)
            sequences_lengths = sequences_lengths.to(device)

            # Skip last batch if it does not have full size
            if sequences.shape[0] < batch_size:
                continue

            base_scores = model(sequences, sequences_lengths)

            losses.append(loss_function(base_scores.view(-1, base_scores.shape[2]),
                                       dot_brackets.view(-1)))
            avg_h_loss, avg_accuracy = compute_metrics(base_scores, dot_brackets)
            h_losses.append(avg_h_loss)
            accuracies.append(avg_accuracy)

        avg_loss = np.mean(losses)
        avg_h_loss = np.mean(h_losses)
        avg_accuracy = np.mean(accuracies)

        print("{} loss: {}".format(mode, avg_loss))
        print("{} hamming loss: {}".format(mode, avg_h_loss))
        print("{} accuracy: {}".format(mode, avg_accuracy))

        return avg_loss, avg_h_loss, avg_accuracy


def evaluate_struct_to_seq(model, test_loader, loss_function, batch_size, mode='test',
                           device='cpu', verbose=False):
    model.eval()
    with torch.no_grad():
        losses = []
        h_losses = []
        accuracies = []

        for batch_idx, (dot_brackets, sequences, sequences_lengths) in enumerate(test_loader):
            dot_brackets = dot_brackets.to(device)
            sequences = sequences.to(device)
            sequences_lengths = sequences_lengths.to(device)

            # Skip last batch if it does not have full size
            if dot_brackets.shape[0] < batch_size:
                continue

            base_scores = model(dot_brackets, sequences_lengths)

            losses.append(loss_function(base_scores.view(-1, base_scores.shape[2]),
                                        dot_brackets.view(-1)).item())
            avg_h_loss, avg_accuracy = compute_metrics(target_dot_brackets=dot_brackets,
                                                       input_sequences=sequences,
                                                       pred_sequences_scores=base_scores,
                                                       sequences_lengths=sequences_lengths,
                                                       verbose=verbose)
            h_losses.append(avg_h_loss)
            accuracies.append(avg_accuracy)

        avg_loss = np.mean(losses)
        avg_h_loss = np.mean(h_losses)
        avg_accuracy = np.mean(accuracies)

        print("{} loss: {}".format(mode, avg_loss))
        print("{} hamming loss: {}".format(mode, avg_h_loss))
        print("{} accuracy: {}".format(mode, avg_accuracy))

        return avg_loss, avg_h_loss, avg_accuracy


def evaluate_struct_to_seq_graph(model, test_loader, loss_function=None, batch_size=None,
                                 mode='test', device='cpu', verbose=False, gan=False,
                                 n_random_features=0):
    model.eval()
    with torch.no_grad():
        losses = []
        h_losses = []
        accuracies = []

        for batch_idx, data in enumerate(test_loader):
            data.x = data.x.to(device)
            data.edge_index = data.edge_index.to(device)
            data.edge_attr = data.edge_attr.to(device)
            data.batch = data.batch.to(device)
            dot_bracket = data.y.to(device)
            sequence = data.sequence.to(device)

            if gan:
                z = torch.Tensor(np.random.normal(0, 1, (data.x.shape[0], n_random_features))).to(
                    device)
                data.x = torch.cat((data.x, z), dim=1)

            pred_sequences_scores = model(data)

            if loss_function:
                losses.append(loss_function(pred_sequences_scores, sequence).item())

            # Metrics are computed with respect to generated folding
            avg_h_loss, avg_accuracy = compute_metrics_graph(target_dot_brackets=dot_bracket,
                                                             input_sequences=sequence,
                                                             pred_sequences_scores=pred_sequences_scores,
                                                             batch=data.batch,
                                                             verbose=verbose)
            h_losses.append(avg_h_loss)
            accuracies.append(avg_accuracy)

        avg_loss = 0 if not losses else np.mean(losses)
        avg_h_loss = np.mean(h_losses)
        avg_accuracy = np.mean(accuracies)

        print("{} loss: {}".format(mode, avg_loss))
        print("{} hamming loss: {}".format(mode, avg_h_loss))
        print("{} accuracy: {}".format(mode, avg_accuracy))

        return avg_loss, avg_h_loss, avg_accuracy


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
