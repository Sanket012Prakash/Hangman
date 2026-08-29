"""
Hangman Solver — Streamlit frontend

Modes:
  1. Random word from test_words.txt — BiLSTM solver plays it out
  2. Custom word input — BiLSTM solver plays it out

After each solve, guesses animate step-by-step (2s per guess) in the board + log.
"""

from __future__ import annotations

import time

import streamlit as st
import streamlit.components.v1 as components

from game import MAX_TRIES
from solver import GameResult, HangmanSolver, load_solver

st.set_page_config(
    page_title="Hangman Solver",
    page_icon="H",
    layout="centered",
    initial_sidebar_state="collapsed",
)

STYLES = """
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;600;700&family=Libre+Baskerville:wght@700&display=swap');

html, body, [class*="css"] {
  font-family: 'DM Sans', sans-serif !important;
  direction: ltr !important;
}

.stApp {
  background:
    radial-gradient(ellipse 120% 80% at 10% 0%, #2a4a3c 0%, transparent 55%),
    radial-gradient(ellipse 90% 70% at 100% 100%, #1a2e28 0%, transparent 50%),
    linear-gradient(165deg, #0f1c18 0%, #152820 45%, #0c1612 100%);
  color: #e8f0ea;
  direction: ltr !important;
}

#MainMenu, footer, header { visibility: hidden; }

.block-container {
  padding-top: 2rem;
  padding-bottom: 3rem;
  max-width: 760px;
  direction: ltr !important;
}

.brand {
  font-family: 'Libre Baskerville', Georgia, serif;
  font-size: clamp(2.2rem, 7vw, 3.2rem);
  font-weight: 700;
  letter-spacing: 0.04em;
  color: #f2ebe0;
  margin: 0 0 0.15rem 0;
  line-height: 1.1;
}

.tagline {
  color: #8fa897;
  font-size: 0.95rem;
  margin: 0 0 1.5rem 0;
}

.stage {
  display: grid;
  grid-template-columns: 1fr 1.35fr;
  gap: 1.25rem;
  align-items: center;
  margin-bottom: 1.25rem;
}

@media (max-width: 640px) {
  .stage { grid-template-columns: 1fr; }
}

.gallows-wrap { display: flex; justify-content: center; }
.gallows-wrap svg {
  width: min(200px, 50vw);
  height: auto;
  filter: drop-shadow(0 8px 24px rgba(0,0,0,0.35));
}

.pattern {
  font-family: 'Libre Baskerville', Georgia, serif;
  font-size: clamp(1.25rem, 4.5vw, 1.85rem);
  letter-spacing: 0.22em;
  color: #f2ebe0;
  margin: 0.4rem 0 0.75rem 0;
  word-break: break-all;
  text-align: center;
  direction: ltr !important;
  unicode-bidi: isolate;
}

.meta {
  display: flex;
  justify-content: center;
  gap: 1.1rem;
  flex-wrap: wrap;
  color: #8fa897;
  font-size: 0.85rem;
}
.meta strong { color: #c5d9cb; font-weight: 600; }

.banner {
  text-align: center;
  padding: 0.85rem 1rem;
  margin: 0.75rem 0 1rem 0;
  border-radius: 4px;
  font-weight: 600;
}
.banner.won {
  background: rgba(62, 140, 98, 0.25);
  color: #9ee0b0;
  border: 1px solid rgba(110, 190, 130, 0.35);
}
.banner.lost {
  background: rgba(160, 70, 55, 0.28);
  color: #f0b4a8;
  border: 1px solid rgba(200, 100, 80, 0.35);
}

.guess-log { direction: ltr !important; unicode-bidi: isolate; text-align: left; }
.guess-row {
  display: flex;
  align-items: center;
  gap: 0.55rem;
  flex-wrap: wrap;
  margin: 0.35rem 0;
  font-size: 0.95rem;
}
.badge {
  display: inline-block;
  min-width: 4.6rem;
  text-align: center;
  padding: 0.18rem 0.55rem;
  border-radius: 4px;
  font-weight: 700;
  letter-spacing: 0.04em;
  font-size: 0.8rem;
}
.badge-hit {
  background: #2f9e5a;
  color: #eafff1;
}
.badge-miss {
  background: #d6453a;
  color: #fff0ee;
}
.guess-meta { color: #a8bdb0; }

.secret-line {
  text-align: center;
  color: #8fa897;
  font-size: 0.9rem;
  margin-bottom: 0.5rem;
  direction: ltr !important;
  unicode-bidi: isolate;
}
.secret-line em { color: #e8f0ea; font-style: normal; font-weight: 600; }

.preview-box {
  margin: 0.4rem 0 0.75rem 0;
  padding: 0.65rem 0.85rem;
  background: rgba(30, 52, 44, 0.75);
  border: 1px solid rgba(143, 168, 151, 0.35);
  border-radius: 8px;
  direction: ltr !important;
  unicode-bidi: isolate;
  text-align: left;
  font-family: 'DM Sans', monospace, sans-serif;
  letter-spacing: 0.12em;
  color: #e8f0ea;
}
.preview-label { color: #8fa897; letter-spacing: normal; font-size: 0.8rem; display: block; margin-bottom: 0.25rem; }

/* Streamlit text area — hard LTR */
div[data-testid="stTextArea"] textarea,
div[data-testid="stTextInput"] input {
  direction: ltr !important;
  unicode-bidi: plaintext !important;
  text-align: left !important;
  background: rgba(30, 52, 44, 0.9) !important;
  color: #e8f0ea !important;
  border: 1px solid rgba(143, 168, 151, 0.4) !important;
  border-radius: 8px !important;
  font-family: Consolas, 'Courier New', monospace !important;
  letter-spacing: 0.08em !important;
}

.stButton > button[kind="primary"] {
  background: #3d7a5a !important;
  border: none !important;
  color: #f2ebe0 !important;
  font-weight: 700 !important;
  border-radius: 8px !important;
}
.stButton > button[kind="primary"]:hover { background: #4a916c !important; }

.stButton > button[kind="secondary"] {
  background: rgba(30, 52, 44, 0.85) !important;
  border: 1px solid rgba(143, 168, 151, 0.4) !important;
  color: #e8f0ea !important;
  border-radius: 8px !important;
}

/* Victory smile animation */
.victory {
  text-align: center;
  margin: 0.5rem 0 1rem 0;
  animation: victory-pop 0.85s cubic-bezier(0.22, 1.2, 0.36, 1) both;
}
.victory svg {
  width: 110px;
  height: 110px;
  filter: drop-shadow(0 10px 28px rgba(62, 180, 110, 0.35));
  animation: victory-bounce 1.4s ease-in-out infinite;
}
.victory-caption {
  color: #9ee0b0;
  font-weight: 700;
  font-size: 1.15rem;
  margin-top: 0.35rem;
  letter-spacing: 0.04em;
}
@keyframes victory-pop {
  0% { opacity: 0; transform: scale(0.4) rotate(-12deg); }
  60% { opacity: 1; transform: scale(1.12) rotate(4deg); }
  100% { opacity: 1; transform: scale(1) rotate(0deg); }
}
@keyframes victory-bounce {
  0%, 100% { transform: translateY(0); }
  50% { transform: translateY(-10px); }
}
</style>
"""

# Injected into parent page so Streamlit's controlled input stays logical LTR
LTR_FIX_JS = """
<script>
(function () {
  const doc = window.parent.document;
  function forceLtr(el) {
    if (!el) return;
    el.setAttribute('dir', 'ltr');
    el.setAttribute('lang', 'en');
    el.style.direction = 'ltr';
    el.style.unicodeBidi = 'plaintext';
    el.style.textAlign = 'left';
  }
  function scan() {
    doc.querySelectorAll('textarea, input[type="text"]').forEach(forceLtr);
    doc.documentElement.setAttribute('dir', 'ltr');
    if (doc.body) doc.body.setAttribute('dir', 'ltr');
  }
  scan();
  new MutationObserver(scan).observe(doc.body, { childList: true, subtree: true });
})();
</script>
"""

GALLOWS_SVG = [
    """<svg viewBox="0 0 200 240" xmlns="http://www.w3.org/2000/svg"><path d="M30 220 H170 M50 220 V30 H120 V50" stroke="#c9b896" stroke-width="6" fill="none" stroke-linecap="round"/></svg>""",
    """<svg viewBox="0 0 200 240" xmlns="http://www.w3.org/2000/svg"><path d="M30 220 H170 M50 220 V30 H120 V50" stroke="#c9b896" stroke-width="6" fill="none" stroke-linecap="round"/><circle cx="120" cy="72" r="22" stroke="#e8dcc8" stroke-width="5" fill="none"/></svg>""",
    """<svg viewBox="0 0 200 240" xmlns="http://www.w3.org/2000/svg"><path d="M30 220 H170 M50 220 V30 H120 V50" stroke="#c9b896" stroke-width="6" fill="none" stroke-linecap="round"/><circle cx="120" cy="72" r="22" stroke="#e8dcc8" stroke-width="5" fill="none"/><path d="M120 94 V150" stroke="#e8dcc8" stroke-width="5" stroke-linecap="round"/></svg>""",
    """<svg viewBox="0 0 200 240" xmlns="http://www.w3.org/2000/svg"><path d="M30 220 H170 M50 220 V30 H120 V50" stroke="#c9b896" stroke-width="6" fill="none" stroke-linecap="round"/><circle cx="120" cy="72" r="22" stroke="#e8dcc8" stroke-width="5" fill="none"/><path d="M120 94 V150 M120 110 L90 135" stroke="#e8dcc8" stroke-width="5" stroke-linecap="round"/></svg>""",
    """<svg viewBox="0 0 200 240" xmlns="http://www.w3.org/2000/svg"><path d="M30 220 H170 M50 220 V30 H120 V50" stroke="#c9b896" stroke-width="6" fill="none" stroke-linecap="round"/><circle cx="120" cy="72" r="22" stroke="#e8dcc8" stroke-width="5" fill="none"/><path d="M120 94 V150 M120 110 L90 135 M120 110 L150 135" stroke="#e8dcc8" stroke-width="5" stroke-linecap="round"/></svg>""",
    """<svg viewBox="0 0 200 240" xmlns="http://www.w3.org/2000/svg"><path d="M30 220 H170 M50 220 V30 H120 V50" stroke="#c9b896" stroke-width="6" fill="none" stroke-linecap="round"/><circle cx="120" cy="72" r="22" stroke="#e8dcc8" stroke-width="5" fill="none"/><path d="M120 94 V150 M120 110 L90 135 M120 110 L150 135 M120 150 L95 190" stroke="#e8dcc8" stroke-width="5" stroke-linecap="round"/></svg>""",
    """<svg viewBox="0 0 200 240" xmlns="http://www.w3.org/2000/svg"><path d="M30 220 H170 M50 220 V30 H120 V50" stroke="#c9b896" stroke-width="6" fill="none" stroke-linecap="round"/><circle cx="120" cy="72" r="22" stroke="#d48474" stroke-width="5" fill="none"/><path d="M120 94 V150 M120 110 L90 135 M120 110 L150 135 M120 150 L95 190 M120 150 L145 190" stroke="#d48474" stroke-width="5" stroke-linecap="round"/></svg>""",
]

VICTORY_SMILE = """
<div class="victory">
  <svg viewBox="0 0 120 120" xmlns="http://www.w3.org/2000/svg" aria-label="Victory smile">
    <circle cx="60" cy="60" r="54" fill="#2f9e5a"/>
    <circle cx="42" cy="48" r="7" fill="#0f1c18"/>
    <circle cx="78" cy="48" r="7" fill="#0f1c18"/>
    <path d="M36 70 Q60 96 84 70" stroke="#0f1c18" stroke-width="7" fill="none" stroke-linecap="round"/>
  </svg>
  <div class="victory-caption">Victory!</div>
</div>
"""

STEP_DELAY_SEC = 2.0
BIDI_MARKS = str.maketrans("", "", "\u200e\u200f\u202a\u202b\u202c\u202d\u202e")


@st.cache_resource(show_spinner=False)
def get_solver() -> HangmanSolver:
    return load_solver()


def pattern_display(pattern: str) -> str:
    return " ".join(pattern)


def normalize_word(raw: str, flip: bool = False) -> str:
    """Strip bidi marks, lowercase; optionally reverse if the field still mirrored."""
    word = (raw or "").translate(BIDI_MARKS).strip().lower()
    # Keep letters only for solving
    word = "".join(ch for ch in word if ch.isalpha())
    if flip:
        word = word[::-1]
    return word


def render_board(result: GameResult, step_idx: int) -> None:
    if step_idx <= 0:
        pattern = "_" * len(result.word)
        tries_left = MAX_TRIES
        wrong = 0
    else:
        step = result.steps[step_idx - 1]
        pattern = step.pattern
        tries_left = step.tries_left
        wrong = MAX_TRIES - tries_left

    svg = GALLOWS_SVG[min(wrong, len(GALLOWS_SVG) - 1)]
    st.markdown(
        f"""
        <div class="secret-line">Secret: <em>{result.word}</em></div>
        <div class="stage">
          <div class="gallows-wrap">{svg}</div>
          <div>
            <div class="meta">
              <span>Tries left <strong>{tries_left}</strong> / {MAX_TRIES}</span>
              <span>Guess <strong>{step_idx}</strong> / {result.n_guesses}</span>
              <span>Length <strong>{len(result.word)}</strong></span>
            </div>
            <p class="pattern">{pattern_display(pattern)}</p>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_victory() -> None:
    st.markdown(VICTORY_SMILE, unsafe_allow_html=True)


def render_result_banner(result: GameResult) -> None:
    if result.result == "WIN":
        render_victory()
        st.markdown(
            f'<div class="banner won">Solved — {result.n_guesses} guesses, '
            f"{result.tries_used} wrong</div>",
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f'<div class="banner lost">Failed — used all {MAX_TRIES} wrong tries '
            f"({result.n_guesses} guesses)</div>",
            unsafe_allow_html=True,
        )


def render_guess_log(result: GameResult, up_to: int) -> None:
    st.markdown("#### Guess log")
    if up_to <= 0:
        st.caption("Waiting for first guess…")
        return

    lines = ['<div class="guess-log">']
    for s in result.steps[:up_to]:
        if s.hit:
            badge = '<span class="badge badge-hit">HIT</span>'
        else:
            badge = '<span class="badge badge-miss">MISS</span>'
        pat = pattern_display(s.pattern)
        lines.append(
            f'<div class="guess-row">'
            f"<strong>{s.n_guess:02d}.</strong> "
            f"<code>'{s.letter}'</code> {badge} "
            f'<span class="guess-meta">{pat} · tries left {s.tries_left}</span>'
            f"</div>"
        )
    lines.append("</div>")
    st.markdown("\n".join(lines), unsafe_allow_html=True)


def start_solve(word: str) -> None:
    solver: HangmanSolver = st.session_state.solver
    with st.spinner(f"BiLSTM computing guesses for '{word}'…"):
        result = solver.play(word)
    st.session_state.result = result
    st.session_state.step_idx = 0
    st.session_state.animating = True


def main() -> None:
    st.markdown(STYLES, unsafe_allow_html=True)
    components.html(LTR_FIX_JS, height=0)

    st.markdown('<p class="brand">Hangman Solver</p>', unsafe_allow_html=True)
    st.markdown(
        '<p class="tagline">BiLSTM + statistical priors guess letter-by-letter '
        f"(max {MAX_TRIES} wrong tries). Live replay: {STEP_DELAY_SEC:.0f}s per guess.</p>",
        unsafe_allow_html=True,
    )

    if "solver" not in st.session_state:
        with st.spinner("Loading BiLSTM models and word priors… (first load may take a minute)"):
            try:
                st.session_state.solver = get_solver()
            except Exception as exc:
                st.error(f"Failed to load solver: {exc}")
                st.stop()

    if "animating" not in st.session_state:
        st.session_state.animating = False
    if "step_idx" not in st.session_state:
        st.session_state.step_idx = 0

    solver: HangmanSolver = st.session_state.solver
    animating = bool(st.session_state.animating)

    st.caption(
        f"Ready — train {len(solver.train_words):,} · "
        f"test {len(solver.test_words):,} · short pool {len(solver.short_words):,}"
    )

    tab_random, tab_custom = st.tabs(["Random test word", "Custom word"])

    with tab_random:
        st.write("Picks a held-out word from `test_words.txt` and lets the model solve it.")
        if st.button(
            "Solve random test word",
            type="primary",
            use_container_width=True,
            disabled=animating,
        ):
            start_solve(solver.random_test_word())
            st.rerun()

    with tab_custom:
        st.write(
            "Type the word **left → right** (English letters only). "
            "Check the preview below — that is exactly what the solver will use."
        )
        # text_area is less prone to RTL caret quirks than text_input on some Windows setups
        custom = st.text_area(
            "Word to solve",
            placeholder="type here: apple",
            label_visibility="collapsed",
            key="custom_word",
            height=68,
            disabled=animating,
        )
        flip = st.checkbox(
            "Flip letters (use if the preview still looks reversed)",
            value=False,
            disabled=animating,
            key="flip_custom",
        )
        word_preview = normalize_word(custom, flip=flip)
        st.markdown(
            f'<div class="preview-box"><span class="preview-label">Solver will use</span>'
            f"{word_preview if word_preview else '—'}</div>",
            unsafe_allow_html=True,
        )

        if st.button(
            "Solve custom word",
            type="primary",
            use_container_width=True,
            disabled=animating,
        ):
            word = word_preview
            if not word:
                st.warning("Enter a word first.")
            elif not (3 <= len(word) <= 30):
                st.warning("Length must be between 3 and 30.")
            elif len(set(word)) <= 1:
                st.warning("Need more than one distinct letter.")
            else:
                start_solve(word)
                st.rerun()

    result: GameResult | None = st.session_state.get("result")
    if not result:
        st.info("Choose a mode above to run the solver.")
        return

    n = result.n_guesses
    step_idx = int(st.session_state.step_idx)
    step_idx = max(0, min(step_idx, n))

    st.divider()
    if animating:
        st.caption(f"Solving live… guess {step_idx} / {n} (next in {STEP_DELAY_SEC:.0f}s)")
    else:
        step_idx = st.slider(
            "Replay step",
            min_value=0,
            max_value=n,
            value=step_idx,
            help="0 = blank board, max = final state",
        )
        st.session_state.step_idx = step_idx

    render_board(result, step_idx)
    if not animating and step_idx == n:
        render_result_banner(result)
    render_guess_log(result, up_to=step_idx)

    if animating:
        if step_idx < n:
            time.sleep(STEP_DELAY_SEC)
            st.session_state.step_idx = step_idx + 1
            st.rerun()
        else:
            st.session_state.animating = False
            render_result_banner(result)
            st.rerun()


if __name__ == "__main__":
    main()
