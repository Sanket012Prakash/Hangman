# Hangman Solver — BiLSTM + Statistical Priors

An AI that **plays Hangman as the guesser**. Given a masked word (e.g. `_ pp _ e`), it
picks the next letter to guess and tries to reveal the whole word within **6 wrong
guesses**. It fuses a **Bidirectional LSTM** letter-prediction model with classical
**statistical language priors** (unigram/bigram frequencies) and an **information-gain**
signal over the remaining candidate words.

On **5,000 held-out test words** (never seen in training), the solver wins **56.2%** of
games, using on average **4.28 / 6** wrong tries.

![Streamlit app](https://img.shields.io/badge/Streamlit-live%20demo-3d7a5a) ![Python](https://img.shields.io/badge/Python-3.10%2B-blue)

---

## How it works

For each turn the solver builds three scores per candidate letter and blends them:

1. **BiLSTM probability** — a 2-layer bidirectional LSTM reads the current masked
   pattern and outputs a probability distribution over `a–z`. Two specialist models are
   used: `model_short` for words ≤ 16 characters and `model_all` for longer words.
2. **Statistical priors** — unigram frequency, position-aware bigram frequency, and a
   conditional prior computed over the set of dictionary words still consistent with the
   revealed pattern.
3. **Information gain** — how much each letter is expected to shrink the candidate set.

The final score is a weighted fusion
`score = α·log(p_bilstm) + (1−α)·(β·statistical + γ·info_gain)`
and the highest-scoring un-guessed letter is played. Very short words (≤ 3 letters) fall
back to a frequency-ordered vowel-first heuristic.

The models are trained with a **curriculum masking** scheme (easy → hard: mask 1 letter,
then up to 4, then up to 6) so the network learns to infer letters from increasingly
sparse patterns.

---

## Project structure

```
Hangman/
├── streamlit_app.py          # Streamlit entry point (run this)
├── hangman/                  # Core package
│   ├── __init__.py
│   ├── game.py               # Hangman rules / word-list loading
│   ├── solver.py             # fused guessing strategy
│   ├── bilstm_numpy.py       # NumPy BiLSTM inference (no TensorFlow on deploy)
│   └── ui.py                 # Streamlit interface
├── models/
│   ├── model_all.weights.h5      # BiLSTM for all word lengths
│   └── model_short.weights.h5    # BiLSTM specialist for words ≤ 16 chars
├── data/
│   ├── train_words.txt           # 97.5% training split
│   ├── test_words.txt            # 2.5% held-out test split
│   └── words_250000_train.txt    # raw source dictionary
├── notebooks/
│   ├── 01_dataset_split.ipynb    # clean + split the dictionary
│   ├── 02_train_bilstm.ipynb     # train model_all & model_short
│   └── 03_evaluate.ipynb         # win-rate evaluation on held-out words
├── .streamlit/config.toml
├── requirements.txt
└── README.md
```

---

## Run locally

```bash
# 1. (optional) create a virtual environment
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

# 2. install dependencies
pip install -r requirements.txt

# 3. launch the app
streamlit run streamlit_app.py
```

Then open the URL Streamlit prints (usually `http://localhost:8501`).

**Modes**
- **Random test word** — picks a held-out word and watches the model solve it live.
- **Custom word** — type any English word and the solver plays it out. A live preview
  shows exactly the word the solver will use.

---

## Deploy to Streamlit Community Cloud

1. Push this repository to GitHub.
2. Go to [share.streamlit.io](https://share.streamlit.io) and sign in with GitHub.
3. **New app** → select your repo/branch → set **Main file path** to `streamlit_app.py`.
4. Deploy. The model weights (~18 MB total) are committed to the repo, so no extra setup
   is needed.

The live app does **not** install TensorFlow. Community Cloud currently defaults to
Python 3.14, and `tensorflow-cpu` has no wheels for that ABI. Inference uses NumPy +
`h5py` against the same Keras `.weights.h5` files. Retrain from the notebooks with a
local TensorFlow install (Python 3.10–3.13).

---

## Tech stack

- **NumPy + h5py** — BiLSTM inference from the saved Keras weights.
- **TensorFlow / Keras** — training only (notebooks; Python 3.10–3.13).
- **Streamlit** — web UI.

## Notes

- The two `.weights.h5` files are Keras **native weights-only** checkpoints and must be
  loaded into the exact architecture (`hidden_dim=256, emb_dim=64, num_layers=2`).
- Training was done on Kaggle (GPU); the notebooks in `notebooks/` reproduce the full
  data-split → train → evaluate pipeline.
