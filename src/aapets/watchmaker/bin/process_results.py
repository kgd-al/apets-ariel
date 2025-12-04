import pprint
from pathlib import Path

pprint.pprint(list(Path("results/watchmaker/").glob("**/evolution.csv")))
