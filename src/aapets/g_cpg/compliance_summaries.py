from pathlib import Path

from matplotlib import pyplot as plt
import pandas as pd

from .config import Config


def compliance_summaries(path: Path):
    print(path)

    fig, ax = plt.subplots()

    for w in path.glob("champion_*.zip"):
        f = w.with_suffix(".trajectory.csv")
        df = pd.read_csv(f)
        print(df)

        ax.plot(df.x, df.y, c=df.t)

    fig.savefig(base.with_suffix(f"merged_trajectories.{args.pl}"))
