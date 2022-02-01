# Minimal code example to access data
# Code source (classes): https://github.com/EML4U/Clustering/tree/9b53ef750861c5ea9f77738dc211067418a99e00



# Class for loading data
import classes.io as iox
io = iox.Io('./')

# Identifiers of dataset
dataset_id = 'amazon-movie-reviews-10000'
descriptor = io.DESCRIPTOR_DOC_TO_VEC
details = 'dim50-epochs50'

# Load data
texts = io.load_data_pair(dataset_id, io.DATATYPE_TEXT)

# print keys of 1-star
#print(texts.get_a().keys())

# print keys of 5-star
#print(texts.get_b().keys())

# print first key of 1-star and of 5-star
#print(texts.get_a_dict_as_lists()[0][0], texts.get_b_dict_as_lists()[0][0])

# print first text of 1-star
print(texts.get_a_dict_as_lists()[1][0])



embeddings = io.load_data_pair(dataset_id, io.DATATYPE_EMBEDDINGS, descriptor, details)

# print first embeddings of 1-star
print(embeddings.get_a_dict_as_lists()[1][0])



details_2dim = 'dim50-epochs50-umap'
embeggings2dim = io.load_data_pair(dataset_id, io.DATATYPE_EMBEDDINGS, descriptor, details_2dim)

# print first 2-dimensional embeddings of 1-star
print(embeggings2dim.get_a_dict_as_lists()[1][0])
