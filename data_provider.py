import numpy as np
from classes.amazon_pickle_reader import AmazonPickleReader

def load_data(data_address):
    data = AmazonPickleReader(data_address)
    features = data.get_all_bow50()["data"][0]
    targets  = np.array(data.get_all_bow50()["data"][1])[:,[1, 4]]
    print(features.shape)
    print(targets.shape)

    d = data.get_bow50(84090, metadata=True)
    print("------------------------------------")
    for i in range(10):
        print(targets[i])
    print("done")
    return features, targets



# Code for loading 1 star 5 star toy data


# from os import sep
# import numpy as np
# from sklearn.model_selection import train_test_split
# from sklearn.ensemble import RandomForestClassifier
# import Uncertainty as unc
# import UncertaintyM as uncM
# import matplotlib.pyplot as plt

# # Class for loading data
# import classes.io as iox
# io = iox.Io('./')

# # Identifiers of dataset
# dataset_id = 'amazon-movie-reviews-10000'
# descriptor = io.DESCRIPTOR_DOC_TO_VEC
# details = 'dim50-epochs50'

# # [My note] a -> 1 star. b -> 5 star. First dimention is the key [0] and the text [1]. Second dimention is the index of documents

# # Load data text
# texts = io.load_data_pair(dataset_id, io.DATATYPE_TEXT)

# # Load data embeddings
# embeddings = io.load_data_pair(dataset_id, io.DATATYPE_EMBEDDINGS, descriptor, details)

# # create the dataset (with targets including the keys)
# class_1 = np.array(embeddings.get_a_dict_as_lists()[1]) # get the embeddings (features) for class 1 star
# class_5 = np.array(embeddings.get_b_dict_as_lists()[1]) # get the embeddings (features) for class 5 star

# target_1_label = np.zeros(len(class_1)).reshape((-1,1))
# target_5_label = np.ones(len(class_5)).reshape((-1,1))
# target_1_key = np.array(embeddings.get_a_dict_as_lists()[0]).reshape((-1,1)) # get the keys (part of target but not the label) for class 1 star
# target_5_key = np.array(embeddings.get_b_dict_as_lists()[0]).reshape((-1,1)) # get the keys for class 5 star

# target_1 = np.concatenate((target_1_label,target_1_key), axis=1)
# target_5 = np.concatenate((target_5_label,target_5_key), axis=1)

# features = np.concatenate((class_1,class_5))
# targets = np.concatenate((target_1,target_5))
