import pandas as pd
import matplotlib.pyplot as plt

def process_tables(tables):
    """Process extracted tables into DataFrames."""
    processed_tables = []
    for table in tables:
        df = pd.DataFrame(table[1:], columns=table[0])
        processed_tables.append(df)
    return processed_tables

def extract_data_for_visualization(tables):
    """Extract data for visualization."""
    # Example: Extract the first table's data
    x = tables[0]["X-axis"].tolist()
    y = tables[0]["Y-axis"].tolist()
    return x, y

def plot_data(x, y):
    """Generate a plot."""
    plt.plot(x, y)
    plt.xlabel("X-axis")
    plt.ylabel("Y-axis")
    plt.title("Experiment Results")
    return plt