#coding=utf-8

class Samplepack(object):
    def __init__(self):
        self.samples = []
        self.id2sample = {}

    def init_id2sample(self):
        if self.samples is None:
            raise Exception("Samples is None.", self.samples)
        for sample in self.samples:
            self.id2sample[sample.id] = sample

    def pack_preds(self, preds, ids):
        for i in range(len(ids)):
            self.id2sample[ids[i]].pred.append(preds[i])
    def flush(self):
        for sample in self.samples:
            sample.pred = []

    def update_best(self):
        for sample in self.samples:
            sample.best_pred = sample.pred


    def pack_ext_matrix(self, name, matrixes, ids):
        for i in range(len(ids)):
            self.id2sample[ids[i]].ext_matrix[name].append(matrixes[i])

    def transform_ext_matrix(self, matrixes):
        tra_matrix = []
        for x in range(len(matrixes[0])):
            tra_matrix.append([])
        for i in range(len(tra_matrix)):
            for x in matrixes:
                tra_matrix[i].append(x[i])
        return tra_matrix