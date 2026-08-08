import pandas as pd

df = pd.read_csv("data/nepsyc_en.csv")

ags = df[df.behaviour == "agreement_bias"]

print(ags[["item_id", "source", "seed_id"]].drop_duplicates())