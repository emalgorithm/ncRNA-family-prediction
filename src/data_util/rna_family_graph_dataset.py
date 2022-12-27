from torch_geometric.data import InMemoryDataset, Data
import pickle
import numpy as np
import torch
from Bio import SeqIO

from src.data_util.data_processing import prepare_sequence
from src.data_util.data_constants import word_to_ix, tag_to_ix, families
from src.util.util import dotbracket_to_graph


class RNAFamilyGraphDataset(InMemoryDataset):
    def __init__(self, file_path, foldings_path, transform=None, pre_transform=None,
                 seq_max_len=10000,
                 seq_min_len=1, n_samples=None):
        super(RNAFamilyGraphDataset, self).__init__(file_path, transform, pre_transform)

        with open(file_path+'/train.fasta', "r") as handle:
            records = list(SeqIO.parse(handle, "fasta"))

        foldings = pickle.load(open(foldings_path, "rb"))

        np.random.shuffle(records)
        records = [x for x in records if seq_min_len <= len(str(x.seq)) <= seq_max_len]
        records = records if not n_samples else records[:n_samples]

        lengths = [len(x) for x in records]

        print("{} sequences found at path {} with max length {}, average length of {}, "
              "and median length of {}".format(len(lengths), file_path, np.max(lengths),
                                               np.mean(lengths), np.median(lengths)))

        data_list = []

        for x in records:
            sequence_string = str(x.seq)
            sequence = prepare_sequence(sequence_string, word_to_ix)

            family = x.description.split()[-1]

            dot_bracket_string = foldings[sequence_string][0]
            dot_bracket = prepare_sequence(dot_bracket_string, tag_to_ix)

            g = dotbracket_to_graph(dot_bracket_string)

            x = sequence

            edges = list(g.edges(data=True))
            # One-hot encoding of the edge type
            edge_attr = torch.Tensor([[0, 1] if e[2]['edge_type'] == 'adjacent' else [1, 0] for e in
                                      edges])
            edge_index = torch.LongTensor(list(g.edges())).t().contiguous()

            y = self.get_family_idx(family)

            data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)

            data_list.append(data)

        self.data, self.slices = self.collate(data_list)

    def download(self):
        pass

    def process(self):
        pass

    @property
    def raw_file_names(self):
        return []

    @property
    def processed_file_names(self):
        return []

    @staticmethod
    def get_family_idx(family):
        if family not in families:
            raise Exception("Family not in list")

        return torch.LongTensor([families.index(family)])
