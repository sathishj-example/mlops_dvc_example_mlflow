import pandas as pd
import os
import yaml
from sklearn.model_selection import train_test_split

def main(params_path):
    with open(params_path) as f:
        params = yaml.safe_load(f)

    p = params['preprocess']
    os.makedirs(os.path.dirname(p['output_train']), exist_ok=True)

    raw = params['generate']['output']
    df = pd.read_csv(raw)

    train, test = train_test_split(df, test_size=p['test_size'], random_state=params['seed'])
    train.to_csv(p['output_train'], index=False)
    test.to_csv(p['output_test'], index=False)
    print(f"Saved train -> {p['output_train']}, test -> {p['output_test']}")

if __name__ == "__main__":
    import sys
    params_path = sys.argv[1]
    main(params_path)
