import pandas as pd


def create_csv(subs, out="out.csv"):
    subjects = [x+" " for x in subs]

    raw_df = pd.read_csv('dataset\crs_ay2223s2.csv')
    df = raw_df[raw_df['Class Name'].str.contains("|".join(subjects))]

    # change class type
    for type, name in enumerate(subjects):
        df.loc[df['Class Name'].str.contains(name), 'Class Type'] = type

    df.to_csv(out, index=False)

if __name__=="__main__":
    subjects = ["EEE 121", "EEE 128", "Math 21", "Fil 40"]
    create_csv(subjects, out="try.csv")
