import networkx as nx
import os
import pickle


def dotbracket_to_graph(dotbracket):
    G = nx.Graph()
    bases = []

    for i, c in enumerate(dotbracket):
        if c == '(':
            bases.append(i)
        elif c == ')':
            neighbor = bases.pop()
            G.add_edge(i, neighbor, edge_type='base_pair')
        elif c == '.':
            G.add_node(i)
        else:
            print("Input is not in dot-bracket notation!")
            return None

        if i > 0:
            G.add_edge(i, i - 1, edge_type='adjacent')
    return G


def get_family_to_sequences():
    family_sequences_path = '../data/family_rna_sequences/'
    rna_family_files = sorted(os.listdir(family_sequences_path))
    family_to_sequences = {}

    for file in rna_family_files:
        if 'RF' in file:
            family = file[:7]
            family_sequences = pickle.load(open(family_sequences_path + file, 'rb'))
            family_to_sequences[family] = family_sequences

    return family_to_sequences
