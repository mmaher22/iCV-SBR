def cau_recall_mrr_org(preds,labels,cutoff = 20):
    recall = []
    mrr = []
    rank_l = []

    for batch, b_label in zip(preds,labels):

        ranks = (batch[b_label] < batch).sum() +1
        rank_l.append(ranks)
        recall.append(ranks <= cutoff)
        mrr.append(1/ranks if ranks <= cutoff else 0.0)
    return recall, mrr, rank_l


def cau_samples_recall_mrr(samples, cutoff=20):
    recall = 0.0
    mrr =0.0
    for sample in samples:
        recall += sum(x <= cutoff for x in sample.pred)
        mrr += sum(1/x if x <= cutoff else 0 for x in sample.pred)
    num = 0
    for sample in samples:
        num += len(sample.pred)
    recall = recall/ num
    mrr = mrr/num
    return recall , mrr