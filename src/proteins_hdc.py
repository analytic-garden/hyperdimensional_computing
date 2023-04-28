#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
proteins_hdc.py - classify proteins using HDC vectors
author: Bill Thompson
license: GPL 3
copyright: 2023-04-24
"""
import numpy as np
from hdc import hdv, cos
from Bio import SeqIO
from Bio.SeqRecord import SeqRecord

def read_fasta(filename: str) -> list[SeqRecord]:
    """read a collection of sequences from a FASTA file.

    Parameters
    ----------
    filename : str
        The file containing sequences

    Returns
    -------
    list[SeqRecord]
        a list of BioPython SeqRecords
    """
    seq_recs = []
    for seq in SeqIO.parse(filename, "fasta"):
        seq_recs.append(seq)

    return seq_recs

def trimer_hdv(amino_acids: list[str], N: int = 10_000) -> dict[str, np.ndarray]:
    """generate hdvs of all 21*21*21 possible amino acid trimers (20 + stop codon)

    Parameters
    ----------
    amino_acids : list[str]
        a list of amino acid symbols
    N : int, optional
        hdv size, by default 10_000

    Returns
    -------
    dict[str, np.ndarray]
        a dictionary, key: amino acid trimer, value: random hdv encoding trimer
    """
    trimer_hdvs = dict()
    for aa1 in amino_acids:
        for aa2 in amino_acids:
            for aa3 in amino_acids:
                trimer_hdvs[aa1 + aa2 + aa3] = hdv(N = N)

    return trimer_hdvs 

def embed_sequences(seqs: list[SeqRecord], 
                    trimer_hdvs: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """Bundle the trimers in each sequence into an hdv

    Parameters
    ----------
    seqs : list[SeqRecord]
        a list of BioPython SeqRecords
    trimer_hdvs : dict[str, np.ndarray]
        a dictionary of trimers created by trimer_hdv

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        an array of sequence hdvs, an array of species for each hdv
    """
    hdvs = np.zeros((len(seqs), len(trimer_hdvs[next(iter(trimer_hdvs))])))
    seq_types = []
    for idx, seq in enumerate(seqs):
        for pos in range(len(seq) - 2):
            hdvs[idx, :] += trimer_hdvs[seqs[idx].seq[pos:(pos+3)]]
        hdvs[idx, :] = np.sign(hdvs[idx, :])
        if seqs[idx].id.find("HUMAN") != -1:
            seq_types.append("HUMAN")
        else:
            seq_types.append("YEAST")

    return hdvs, np.array(seq_types, dtype = str)

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

def prototype(hdvs: np.ndarray, idx: list) -> np.ndarray:
    """Build a protype hdv by bundling an array of hdvs

    Parameters
    ----------
    hdvs : np.ndarray
        an array of hdvs
    idx : list
        a list of inices into hdvs

    Returns
    -------
    np.ndarray
        an hdv created by bundling the choses hdvs

    Requires
    --------
    All elements of idx must be >= 0 and < hdvs.shape[0].
    """
    return np.sign(np.sum(hdvs[idx], axis = 0))

def prediction(human_prototype: np.ndarray,
               yeast_prototype: np.ndarray,
               x: np.ndarray) -> str:
    """Determine whether the hdv, x, is more likely to be human or yeast

    Parameters
    ----------
    human_prototype : np.ndarray
        an hdv of human sequences created by prototype
    yeast_prototype : np.ndarray
        an hdv of yeast sequences created by prototype
    x : np.ndarray
        the query hdv

    Returns
    -------
    str
        Either HUMAN or YEAST depending on which x is closer to
    """
    return "HUMAN" if cos(human_prototype, x) > cos(yeast_prototype, x) else "YEAST"

def main():
    data_file = "data/sapiens_yeast_proteins.fasta"
    amino_acids = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'Y']

    seq_recs= read_fasta(data_file)
    trimer_hdvs = trimer_hdv(amino_acids)

    hdvs, seq_types = embed_sequences(seq_recs, trimer_hdvs)
    training_idx, test_idx = get_training_test_index(hdvs.shape[0])

    human_training_idx = [i for i in training_idx if seq_types[i] == "HUMAN"]
    human_prototype = prototype(hdvs, human_training_idx)
    yeast_training_idx = [i for i in training_idx if seq_types[i] == "YEAST"]
    yeast_prototype = prototype(hdvs, yeast_training_idx)

    predictions = np.array([prediction(human_prototype, yeast_prototype, x) for x in hdvs[test_idx]], dtype = str)

    print(np.mean(predictions == seq_types[test_idx]))

    print()

if __name__ == "__main__":
    main()
