# coding:utf-8
from __future__ import print_function

import numpy
numpy.random.seed(42)


def prepare_data(seqs, labels):
    """Create the matrices from the datasets.

    This pad each sequence to the same lenght: the lenght of the
    longuest sequence or maxlen.

    if maxlen is set, we will cut all sequence to this maximum
    lenght.

    This swap the axis!
    """
    # x: a list of sentences

    lengths = [len(s) for s in seqs]
    n_samples = len(seqs)
    maxlen = numpy.max(lengths)
    x = numpy.zeros((n_samples, maxlen), dtype=numpy.int64)
    x_mask = numpy.ones((n_samples, maxlen), dtype=numpy.float32)
    for idx, s in enumerate(seqs):
        x[idx, :lengths[idx]] = s

    x_mask *= (1 - (x == 0))
    # seq_length = [i if i <= maxlen else maxlen for i in lengths]

    return x, x_mask, labels, lengths

