from algorithms import Clustera_KMeans
from datasets import Leukemia10_Dataset_Builder

import pandas as pd
import numpy as np

np.random.seed(101)

# df = pd.DataFrame(data=np.random.randint(0, 6, 20).reshape(5, 4), index='A B C D E'.split(), columns='W X Y Z'.split())

model = Clustera_KMeans(n_clusters=2)



leukemia = Leukemia10_Dataset_Builder()
leukemia.download_prepare()
df = leukemia.as_dataframe()

data = df.drop('class', axis=1)

model.fit(data)

model.accuracy(leukemia)