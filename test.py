from collections import Counter
import pandas as pd
a = Counter({'menu': 20, 'good': 15, 'tits': 85, 'bar': 5})
b = Counter({'menu': 1, 'good': 1, 'bar': 3})

print(a["menu"])


# a_df = pd.DataFrame.from_dict(a, orient='index').reset_index()
# b_df = pd.DataFrame.from_dict(b, orient='index').reset_index()

# print(a_df)
# print("------------------------------------")
# print(b_df)
# print("------------------------------------")
# print("------------------------------------")
# c_df = a_df + b_df
# print(c_df)