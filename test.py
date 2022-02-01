import data_provider as dp

features, targets = dp.load_data("./Data/")
features_list, targets_list = dp.partition_data(features, targets,10)
