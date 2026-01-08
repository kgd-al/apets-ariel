import pprint
from pathlib import Path

import pandas as pd
import seaborn as sns


def process(path: Path):
    _df = pd.read_csv(path, sep=' ', usecols=range(4))
    # This gives us the champion, even with anil's old data
    if "anil" in str(path):
        print(_df.to_string(min_rows=30))
    _df = _df.iloc[5:].iloc[::5]
    _df["Type"] = path.parts[-3]
    _df["Run"] = path.parts[-2]
    _df["Evaluations"] = _df.index
    if "anil" in str(path):
        print(_df.to_string(min_rows=30))
        _df.Speed /= 100
    return _df


df = pd.concat([
    process(path) for path
    in Path("results/watchmaker/").glob("**/evolution.csv")
])

g = sns.lineplot(data=df, x="Evaluations", y="Speed", hue="Type")
g.figure.savefig("watchmaker.pdf", bbox_inches="tight")
