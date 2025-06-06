import pandas as pd
import matplotlib.pyplot as plt

def plot_product_types(predictions):
    df = pd.DataFrame(predictions['products'])
    df['type'].value_counts().plot(kind='bar', color='cornflowerblue')
    plt.title('Predicted Product Types')
    plt.xlabel('Type')
    plt.ylabel('Count')
    plt.show()

def plot_confidence_histogram(predictions):
    df = pd.DataFrame(predictions['products'])
    df['confidence'].hist(bins=10, color='darkseagreen')
    plt.title('CLIP Match Confidence Scores')
    plt.xlabel('Confidence')
    plt.ylabel('Num Predictions')
    plt.show()

def plot_vibe_distribution(outputs):
    import itertools
    all_vibes = list(itertools.chain.from_iterable([out['vibes'] for out in outputs]))
    pd.Series(all_vibes).value_counts().plot(kind='bar', color='orchid')
    plt.title("Vibe Distribution Across Videos")
    plt.xlabel("Vibe")
    plt.ylabel("Count")
    plt.show()
