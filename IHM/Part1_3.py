import pandas as pd
import numpy as np
from itertools import combinations
import math

def generate_labels(name, number):
    return [name+"_"+str(i+1) for i in range(number)]

def equal_width(data, num_bins, labels=None, include_lowest=False, right=True):
    data_min, data_max = min(data), max(data)
    bin_width = (data_max - data_min) / num_bins
    bin_edges = [data_min + i * bin_width for i in range(num_bins + 1)]
    if labels is None:
        labels = [f'Interval {i+1}' for i in range(num_bins)]
    bin_labels = []
    for value in data:
        for i in range(len(bin_edges) - 1):
            if (include_lowest and i == 0) or (right and value <= bin_edges[i + 1] and value > bin_edges[i]) or (
                    not right and value < bin_edges[i + 1] and value >= bin_edges[i]):
                bin_labels.append(labels[i])
                break
        else:
            bin_labels.append(labels[-1])

    return bin_labels

def equal_freq(data, q, labels):
    quantiles = np.percentile(data, np.linspace(0, 100, q+1))
    labeled_data = np.digitize(data, quantiles)
    labeled_series = pd.Series(labeled_data, name=data.name)
    labeled_series.replace(range(1, q + 1), labels, inplace=True)
    return labeled_series

def get_k_itemsets(data, k):
    itemsets = []
    for row in data:
        itemsets.extend(combinations(row, k))
    return itemsets

def calculate_support(data, itemset):
    count = 0
    for row in data:
        if all(item in row for item in itemset):
            count += 1
    return count / len(data)

def find_frequent_itemsets(data, min_support):
    itemsets = get_k_itemsets(data, 1)
    frequent_itemsets = []

    k = 1
    while itemsets:
        frequent_itemsets_k = []

        for itemset in itemsets:
            support = calculate_support(data, itemset)
            if support >= min_support:
                frequent_itemsets_k.append((itemset, support))

        frequent_itemsets.extend(frequent_itemsets_k)

        k += 1
        itemsets = list(set(combinations(set(item for itemset in frequent_itemsets_k for item in itemset[0]), k)))

    return frequent_itemsets

def generate_rules(frequent_itemsets, min_confidence):
    rules = []

    for itemset, support in frequent_itemsets:
        if len(itemset) > 1:
            for i in range(1, len(itemset)):
                antecedent = itemset[:i]
                consequent = itemset[i:]

                support_antecedent = calculate_support(df.values.tolist(), antecedent)
                support_consequent = calculate_support(df.values.tolist(), consequent)
                confidence = support / support_antecedent
                rule_support = support
                if confidence >= min_confidence:
                    rules.append((antecedent, support_antecedent, consequent, support_consequent, rule_support, confidence))

    return rules

def correlation(rules):
    rules_with_correlation = []
    for antecedent, support_antecedent, consequent, support_consequent, rule_support, confidence in rules:
        lift = rule_support/(support_antecedent*support_consequent)
        confidence = rule_support/max(support_antecedent, support_consequent)
        cosine = rule_support/math.sqrt(support_antecedent*support_consequent)

        rules_with_correlation.append((antecedent, consequent, lift, confidence, cosine))
    return rules_with_correlation