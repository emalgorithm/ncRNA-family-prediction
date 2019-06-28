# 'U' and 'T' in this sequences refer both to the base 'U'. 'T' is just used for convenience
word_to_ix = {"<PAD>": 0, "A": 1, "G": 2, "C": 3, "T": 4, "N": 5, "M": 6, "R": 7, "Y": 8, "W": 9,
              "K": 10, "S": 11, "H": 12, "V": 13, "B": 14, "D": 15}
ix_to_word = {v: k for k, v in word_to_ix.items()}

tag_to_ix = {"<PAD>": 0, ".": 1, "(": 2, ")": 3}
ix_to_tag = {v: k for k, v in tag_to_ix.items()}

families = [
           '5S_rRNA',
           '5_8S_rRNA',
           'CD-box',
           'HACA-box',
           'IRES',
           'Intron_gpI',
           'Intron_gpII',
           'leader',
           'miRNA',
           'riboswitch',
           'ribozyme',
           'scaRNA',
           'tRNA'
           ]

