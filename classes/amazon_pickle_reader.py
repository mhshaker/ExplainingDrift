# Reader for preprocessed Amazon movie reviews
#
# The preprocessed data was sorted by time and rounded by hour [1], resulting in new identifiers "raw_id".
# The BoW-50 data does not contain 84,090 entries, mainly (by rounding) from reviews between 1997 and 1999.
#
# Preprocessed data: https://drive.google.com/drive/folders/1NdfbAkH-YRpHul4uwsIN3_O5T_VQmGY1
# Original data: https://snap.stanford.edu/data/web-Movies.html
# Data was preprocessed using
# [1] https://github.com/EML4U/Clustering/blob/1bebd2e9febec60245cc2ef66a5324b9dcd9497e/amazon_movie_sorter.py#L37
# [2] https://github.com/EML4U/Clustering/blob/1bebd2e9febec60245cc2ef66a5324b9dcd9497e/generator_amazon_movie_embeddings.py#L67
#
# Example:
# data = AmazonPickleReader('/home/eml4u/EML4U/data/amazon-complete/')
# print(data.get_text(84090))
# print(data.get_bow50(84090))
# print(data.get_text(84090, metadata=True))
# print(data.get_bow50(84090, metadata=True))

import pickle

class AmazonPickleReader:
    
    def __init__(self, data_directory):
        self.data_directory = data_directory
        self.filename_raw   = 'amazon_raw.pickle'
        self.filename_bow50 = 'amazon_bow_50.pickle'
        self.data_raw   = None
        self.data_bow50 = None

    def set_filename_raw(filename):
        self.filename_raw = filename
        
    def set_filename_bow_50(filename):
        self.filename_bow50 = filename
        
    def get_all_raw(self):
        if(self.data_raw is None):
            with open(self.data_directory + self.filename_raw, 'rb') as handle:
                self.data_raw = pickle.load(handle)
        return self.data_raw
    
    def get_all_bow50(self):
        if(self.data_bow50 is None):
            with open(self.data_directory + self.filename_bow50, 'rb') as handle:
                self.data_bow50 = pickle.load(handle)
        return self.data_bow50

    def get_text(self, raw_id, metadata=False):
        if metadata:
            return self.get_all_raw()[1][raw_id]
        else:
            return self.get_all_raw()[0][raw_id]
    
    def get_bow50(self, raw_id, metadata=False):
        # 1997 to 1999 not included
        bow50_id = raw_id - 84090
        if(bow50_id < 0):
            raise IndexError('list index out of range: ' + str(raw_id))
        if metadata:
            return self.get_all_bow50()["data"][1][bow50_id]
        else:
            return self.get_all_bow50()["data"][0][bow50_id]