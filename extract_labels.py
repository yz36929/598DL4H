import pandas as pd

dx = pd.read_csv("data/adni_clinical/DXSUM_PDXCONV_ADNIALL.csv", low_memory=False)
dx = dx[["PTID","VISCODE","DXCURREN"]] \
       .rename(columns={"PTID":"subject_id","VISCODE":"session","DXCURREN":"dxcode"})

# Map the numeric codes to labels:
#   1=CN, 2=MCI, 3=AD (please double‐check in your DXSUM docs)
label_map = {1:"CN", 2:"MCI", 3:"AD"}
dx["diagnosis"] = dx["dxcode"].map(label_map)

# Bring in age from ADNIMERGE
am = pd.read_csv("data/adni_clinical/ADNIMERGE.csv", low_memory=False)
am = am[["PTID","VISCODE","AGE"]].rename(columns={
    "PTID":"subject_id","VISCODE":"session","AGE":"age"
})

# Merge them
df = pd.merge(am, dx[["subject_id","session","diagnosis"]],
              on=["subject_id","session"], how="inner")

# Keep only CN/MCI/AD and save
df = df[df["diagnosis"].isin(["CN","MCI","AD"])] \
       .drop_duplicates(["subject_id","session"]) \
       .sort_values(["subject_id","session"])
df.to_csv("data/labels.tsv", sep="\t", index=False)
print("Wrote", len(df), "rows to labels.tsv")
