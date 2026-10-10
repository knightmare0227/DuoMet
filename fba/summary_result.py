import os

import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support, average_precision_score

import read_paras


def compute_confusion_matrix(gem):
    r = gem.loc[:, ["phenotype__no_gapfill", "phenotype__w_gapfill"]]
    L0 = r[r["phenotype__no_gapfill"] == 0]
    TN = len(L0[L0["phenotype__w_gapfill"] == 0])
    FP = len(L0[L0["phenotype__w_gapfill"] == 1])
    L1 = r[r["phenotype__no_gapfill"] == 1]
    TP = len(L1[L1["phenotype__w_gapfill"] == 1])
    FN = len(L1[L1["phenotype__w_gapfill"] == 0])
    return TP, FP, TN, FN


def compute_indexes(TP, FP, TN, FN):
    accuracy = (TP + TN) / (TP + TN + FP + FN)  # 准确率
    precision = TP / (TP + FP)  # 精确率
    recall = TP / (TP + FN)  # 召回率
    F1 = (2 * precision * recall) / (precision + recall)  # F1
    return accuracy, precision, recall, F1


def main(gems_list):
    input_file = 'input_parameters.txt'

    cm_list = []
    cm_results = pd.DataFrame([], columns=['model', 'gem', 'TP', 'FP', 'TN', 'FN'])
    results = []

    for gem_name in gems_list:
        gem_name = gem_name[:-4]
        # print(f'gems:{i}')
        paras = read_paras.read(input_file, gem_name,'200')
        output_file: str = "%s/%s" % (paras['OUTPUT_DIRECTORY'], paras['OUTPUT_FILENAME'])
        gem = pd.read_csv(output_file)

        cheshire_precited = np.array(gem.loc[:, "phenotype__w_gapfill"])
        cheshire_label = np.array(gem.loc[:, "phenotype__no_gapfill"])
        TP, FP, TN, FN = compute_confusion_matrix(gem)
        cm_results = pd.concat(
            [cm_results,
             pd.DataFrame([{'model': paras['MODEL'], 'gem': gem_name, 'TP': TP, 'FP': FP, 'TN': TN, 'FN': FN}])],
            axis=0)
        # report = precision_recall_fscore_support(cheshire_label, cheshire_precited, labels=[1])

        # print(f':gem:{i}, TP:{TP}, FP:{FP}, TN:{TN}, FN:{FN}')

        if TP == 0:
            cheshire_accuracy = 0
            cheshire_precision = 0
            cheshire_recall = 0
            cheshire_F1 = 0
            auc = 0
            aupr = 0
        else:
            cheshire_accuracy, cheshire_precision, cheshire_recall, cheshire_F1 = compute_indexes(TP, FP, TN, FN)
            auc = roc_auc_score(cheshire_label, cheshire_precited)
            aupr = average_precision_score(cheshire_label, cheshire_precited)

        gem_name = gem_name.replace('_', '.')
        model = paras['MODEL']
        results.append(pd.DataFrame(
            [{'model': paras['MODEL'], 'gem': gem_name, f'{model}.F1': cheshire_F1, f'{model}.AUROC': auc,
              f'{model}.AUPR': aupr, f'{model}.Accuracy': cheshire_accuracy,
              f'{model}.Recall': cheshire_recall,
              f'{model}.Precision': cheshire_precision}]))
        print(
            f'{paras["MODEL"]}: f1: {cheshire_F1}, auc: {auc}, aupr: {aupr}, acc: {cheshire_accuracy}, re:{cheshire_recall},pr:{cheshire_precision}')
    results = pd.concat(results)
    results.to_csv('/'.join(paras['OUTPUT_DIRECTORY'].split('/')[:-1]) + f"/{paras['MODEL']}_ft_summary.csv",
                   index=False)


if __name__ == '__main__':
    root_path = '../data/gems/'
    data_list = sorted(os.listdir(root_path))
    
    main(data_list)
