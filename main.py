from os.path import join, dirname

import pandas as pd


path = join(dirname(__file__), "data", "samples.csv")
data = pd.read_csv(path, sep=",")
print(data.head())

X = list(data["text"])
y = list(data["label"])

X = TfidfVectorizer.fit_transform(X)
y = LabelEncoder.fit(y)
