"""
helper util to calculate dataset lengths
"""
import numpy as np


def get_dataset_lengths(dataset):
    if "length" in dataset.data.column_names:
        lengths = np.array(dataset["length"])
    elif "position_ids" in dataset.data.column_names:
        position_ids = dataset["position_ids"]
        lengths = np.array([x[-1] + 1 for x in position_ids])
    else:
        input_ids = dataset["input_ids"]
        lengths = np.vectorize(len)(np.array(input_ids, dtype=object))
        return lengths
    return lengths
