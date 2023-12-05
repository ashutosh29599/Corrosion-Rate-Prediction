from IPython.display import display

from data_loader import load_data

df = load_data("corrosion data combined.csv")

display(df.describe())
df.describe().to_csv("data_stats.csv")
