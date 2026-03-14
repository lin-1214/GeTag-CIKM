import json
import matplotlib.pyplot as plt
import pandas as pd

def plot_best_metrics(file_path='json/iteration_metrics.json'):

    df = pd.read_csv('json/iteration_prompt_scores.csv')
    df['best_score'] = pd.to_numeric(df['best_score'], errors='coerce')
    best_per_iter = (df
        .groupby('iteration', as_index=False)['best_score']
        .max()
        .sort_values('iteration'))

    plt.figure(figsize=(7,4))
    plt.plot(best_per_iter['iteration'], best_per_iter['best_score'], marker='o')
    plt.xlabel('Iteration')
    plt.ylabel('Best score (ndcg@10/100/test)')
    plt.title('Per-iteration Best Score')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('json/iteration_best_scores.png', dpi=150)
    print('Saved to json/iteration_best_scores.png')

if __name__ == '__main__':
    # Ensure you have matplotlib installed: pip install matplotlib
    plot_best_metrics()
