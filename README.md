# Hangman Solver

https://gamehangman.streamlit.app/

BiLSTM + statistical priors. 6 wrong tries max. 56.2% win rate on 5,000 held-out words.

## How it works

Each guess mixes three signals:

1. BiLSTM letter probabilities (`model_short` for words ≤ 16, `model_all` for longer)
2. Unigram / bigram / candidate-set priors
3. Information gain over remaining dictionary words

```
score = α·log(p_bilstm) + (1−α)·(β·statistical + γ·info_gain)
```

Words of 3 letters or fewer just guess vowels first.

## Run locally

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
source .venv/bin/activate       # macOS / Linux

pip install -r requirements.txt
streamlit run streamlit_app.py
```

## Layout

```
streamlit_app.py
hangman/
  game.py
  solver.py
  bilstm_numpy.py
  ui.py
models/
  model_all.weights.h5
  model_short.weights.h5
data/
  train_words.txt
  test_words.txt
  words_250000_train.txt
notebooks/
  01_dataset_split.ipynb
  02_train_bilstm.ipynb
  03_evaluate.ipynb
```
