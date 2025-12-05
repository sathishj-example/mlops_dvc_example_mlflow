import os
import pandas as pd
from sklearn.datasets import make_classification
import yaml

def main(params_path):
    with open(params_path) as f:
        params = yaml.safe_load(f)

    g = params['generate']
    os.makedirs(os.path.dirname(g['output']), exist_ok=True)

    X, y = make_classification(
        n_samples=g['n_samples'],
        n_features=g['n_features'],
        n_informative=g['n_informative'],
        random_state=params['seed']
    )
    df = pd.DataFrame(X, columns=[f"f{i}" for i in range(X.shape[1])])
    df['target'] = y
    df.to_csv(g['output'], index=False)
    print(f"Generated data -> {g['output']}")

if __name__ == "__main__":
    import sys
    params_path = sys.argv[1]
    main(params_path)
