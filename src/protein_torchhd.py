#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
protein_torchhd.py - classify proteins using torchhd
author: Bill Thompson
license: GPL 3
copyright: 2023-05-13
"""
import numpy as np
from Bio import SeqIO
from Bio.SeqRecord import SeqRecord

import torch

import torchhd

class Encoder():
    """A class to encode sequences as torchhd hypervectors

    """
    _amino_acids = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 
                    'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'Y']

    def __init__(self, dim: int = 10_000):
        self._dim = dim
        self._trimers = self._trimer_hdv()  # initialize all possible trimers

    def encode(self, seq: SeqRecord) -> torchhd.tensors.map.MAPTensor:
        """Encode a BioPython SeqRecord as a torchhd tensor hypervector
            The sequence is broken into overlapping 3-mers. Each 3-mer is encoded as a hypervector.
            Hypervectors are bound and quantized.

        Parameters
        ----------
        seq : SeqRecord
            a BioPython sequence record

        Returns
        -------
        torchhd.tensors.map.MAPTensor
            a quantized hypervector 
        """
        hdv = torchhd.empty(len(seq) - 2, self._dim) # a hypervector for each 3-mer
        idx = 0
        for pos in range(len(seq) - 2):
            hdv[idx, :] = self._trimers[seq.seq[pos:(pos+3)]]
            idx += 1

        return torchhd.hard_quantize(torchhd.multiset(hdv))
            
    def _trimer_hdv(self) -> dict[str, torchhd.tensors.map.MAPTensor]:
        """generate hdvs of all 21*21*21 possible amino acid trimers (20 + stop codon)

        Returns
        -------
        dict[str, np.ndarray]
            a dictionary, key: amino acid trimer, value: random hdv encoding trimer
        """
        trimer_hdvs = dict()
        for aa1 in self._amino_acids:
            for aa2 in self._amino_acids:
                for aa3 in self._amino_acids:
                    trimer_hdvs[aa1 + aa2 + aa3] = torchhd.random(1, self._dim)

        return trimer_hdvs 
        
def read_fasta(filename: str) -> tuple[list[SeqRecord], list[int]]:
    """read a collection of sequences from a FASTA file.

    Parameters
    ----------
    filename : str
        The file containing sequences

    Returns
    -------
    tuple
    list[SeqRecord]
        a list of BioPython SeqRecords
    list[str]
        a list of species labels
    """
    seq_recs = []
    seq_types = []
    for seq in SeqIO.parse(filename, "fasta"):
        if seq.id.find("HUMAN") != -1:
            seq_types.append("HUMAN")
        else:
            seq_types.append("YEAST")
        seq_recs.append(seq)

    return seq_recs, seq_types

def get_training_test_index(length: int, training_pct: float = 0.8) -> tuple[list, list]:
    """Create indices of sequence hdvs for training and testing

    Parameters
    ----------
    length : int
        dimension 0 of the hdv array of sequences
    training_pct : float, optional
        proportion of data used for training, by default 0.8

    Returns
    -------
    tuple[list, list]
        list of indices for training and testing 
    """
    idx = np.random.choice(length, size = int(np.rint(training_pct * length)), replace = False)
    mask=np.full(length, True, dtype = bool)
    mask[idx] = False

    ids = np.array(list(range(length)), dtype = int)

    return list(ids[~mask]), list(ids[mask])

def prediction(human_prototype: torchhd.tensors.map.MAPTensor,
               yeast_prototype: torchhd.tensors.map.MAPTensor,
               x: torchhd.tensors.map.MAPTensor) -> str:
    """Determine whether the hdv, x, is more likely to be human or yeast

    Parameters
    ----------
    human_prototype : torchhd.tensors.map.MAPTensor
        an hdv of human sequences 
    yeast_prototype : torchhd.tensors.map.MAPTensor
        an hdv of yeast sequences 
    x : torchhd.tensors.map.MAPTensor
        the query hdv

    Returns
    -------
    str
        Either HUMAN or YEAST depending on which x is closer to
    """
    return "HUMAN" if torchhd.cosine_similarity(human_prototype, x) > torchhd.cosine_similarity(yeast_prototype, x) else "YEAST"

def main():
    dim = 10_000
    data_file = "data/sapiens_yeast_proteins.fasta"

    seq_recs, seq_species = read_fasta(data_file)
    training_idx, test_idx = get_training_test_index(len(seq_recs))  # split the data

    encoder = Encoder(dim = dim)

    # build a prototype hdv for each species by bundling like sequences
    yeast_prototype = torchhd.empty(1, dim)
    human_prototype = torchhd.empty(1, dim)
    for i in training_idx:
        hdv = encoder.encode(seq_recs[i])
        if seq_species[i] == "HUMAN":
            human_prototype = torchhd.bundle(human_prototype, hdv)
        else:
            yeast_prototype = torchhd.bundle(yeast_prototype, hdv) 

    # shrink the dimensions to match encoding and quantize
    yeast_prototype = torch.squeeze(torchhd.hard_quantize(yeast_prototype))
    human_prototype = torch.squeeze(torchhd.hard_quantize(human_prototype))

    # test predictions and print accuracy
    predictions = []
    for i in test_idx:
        hdv = encoder.encode(seq_recs[i])
        predictions.append(prediction(human_prototype, yeast_prototype, hdv))
        
    print(np.mean(np.array(predictions, dtype = str) == np.array(seq_species, dtype = str)[test_idx]))

if __name__ == "__main__":
    main()
