from rouge_score import rouge_scorer
from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.bleu_score import SmoothingFunction
from collections import Counter
import numpy as np

def bleu(hyps, refs):
    """ Calculate bleu 1/2. """
    bleu_1 = []
    bleu_2 = []
    for hyp, ref in zip(hyps, refs):
        try:
            score = sentence_bleu(
                [ref], hyp,
                smoothing_function=SmoothingFunction().method7,
                weights=[1, 0, 0, 0])
        except:
            score = 0
        bleu_1.append(score)
        try:
            score = sentence_bleu(
                [ref], hyp,
                smoothing_function=SmoothingFunction().method7,
                weights=[0.5, 0.5, 0, 0])
        except:
            score = 0
        bleu_2.append(score)
    bleu_1 = np.average(bleu_1)
    bleu_2 = np.average(bleu_2)
    return bleu_1, bleu_2

def distinct(seqs):
    """ Calculate intra/inter distinct 1/2. """
    batch_size = len(seqs)
    intra_dist1, intra_dist2 = [], []
    unigrams_all, bigrams_all = Counter(), Counter()
    for seq in seqs:
        unigrams = Counter(seq)
        bigrams = Counter(zip(seq, seq[1:]))
        intra_dist1.append((len(unigrams)+1e-12) / (len(seq)+1e-5))
        intra_dist2.append((len(bigrams)+1e-12) / (max(0, len(seq)-1)+1e-5))

        unigrams_all.update(unigrams)
        bigrams_all.update(bigrams)

    inter_dist1 = (len(unigrams_all)+1e-12) / (sum(unigrams_all.values())+1e-5) #use this
    inter_dist2 = (len(bigrams_all)+1e-12) / (sum(bigrams_all.values())+1e-5) #use this
    intra_dist1 = np.average(intra_dist1)
    intra_dist2 = np.average(intra_dist2)
    # print(intra_dist1, intra_dist2, inter_dist1, inter_dist2)
    return intra_dist1, intra_dist2, inter_dist1, inter_dist2

def rouge(hyps, refs):
    scorer = rouge_scorer.RougeScorer(rouge_types=['rouge1', 'rouge2', 'rougeL', 'rougeLsum'], use_stemmer=False)
    rouge1 = rouge2 = []
    for ref, pred in zip(refs, hyps):
        score = scorer.score(ref, pred)
        rouge1.append(score['rouge1'].fmeasure)
        rouge2.append(score['rouge2'].fmeasure)
    rouge1 = np.average(rouge1)
    rouge2 = np.average(rouge2)
    return rouge1,rouge2