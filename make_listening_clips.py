"""Build a blind listening test: same songs, different models, labels hidden.

    python make_listening_clips.py

Picks songs across the whole range of scores (best, middle, worst, plus songs with no
guitar at all as a trap), separates each with every model, and writes 20-second clips as
ordinary 16-bit WAVs that any browser can play. For each song the models are shuffled into
anonymous slots A and B, so you cannot tell which is which while listening; the answer key
is written separately and is not read by the page.

Output: runs/listening_test/index.html (open it in a browser), clips/, key.json.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import random

import numpy as np

SAMPLE_RATE = 44100
MODELS = ("htdemucs", "ensemble")     # the choice that matters: baseline vs blend


def pick_songs(rows: dict, counts=(3, 4, 3)) -> list[str]:
    """Best, middle and worst songs by improvement, so the test is not all easy cases."""
    ranked = sorted(rows, key=lambda s: rows[s]["guitar_improvement_db"])
    best, middle, worst = counts[2], counts[1], counts[0]
    picks = ranked[:worst]
    mid = len(ranked) // 2
    picks += ranked[mid - middle // 2: mid - middle // 2 + middle]
    picks += ranked[-best:]
    return list(dict.fromkeys(picks))


def loudest_window(reference: np.ndarray, seconds: float, sample_rate: int = SAMPLE_RATE) -> int:
    length = int(seconds * sample_rate)
    mono = reference.mean(axis=0) if reference.ndim == 2 else reference
    if len(mono) <= length:
        return 0
    energy = np.concatenate([[0.0], np.cumsum(mono.astype(np.float64) ** 2)])
    starts = np.arange(0, len(mono) - length + 1, sample_rate)
    return int(starts[np.argmax(energy[starts + length] - energy[starts])])


def assign_slots(songs, models, seed: int = 7) -> dict:
    """Give each song its own shuffled A/B labelling, so order carries no information."""
    rng = random.Random(seed)
    key = {}
    for song in songs:
        shuffled = list(models)
        rng.shuffle(shuffled)
        key[song] = {chr(ord("A") + i): model for i, model in enumerate(shuffled)}
    return key


def write_page(out: Path, items: list) -> None:
    payload = json.dumps(items, indent=1)
    html = """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><title>SoundClipper listening test</title>
<style>
 :root{color-scheme:light dark}
 body{font:16px/1.5 system-ui,sans-serif;max-width:56rem;margin:0 auto;padding:24px 16px 96px}
 h1{font-size:1.5rem;margin:0 0 .25rem} .sub{opacity:.7;margin:0 0 2rem}
 .item{border:1px solid #8884;border-radius:10px;padding:16px;margin-bottom:20px}
 .item h2{font-size:1.05rem;margin:0 0 .5rem} .row{display:flex;gap:12px;flex-wrap:wrap;align-items:center;margin:.4rem 0}
 audio{height:34px} label{font-size:.95rem}
 .choice button{font:inherit;padding:.4rem .9rem;border-radius:8px;border:1px solid #8886;background:#8881;cursor:pointer}
 .choice button[aria-pressed=true]{background:#2d7;border-color:#2d7;color:#000}
 .stars button{font:inherit;padding:.3rem .6rem;border-radius:6px;border:1px solid #8886;background:#8881;cursor:pointer}
 .stars button[aria-pressed=true]{background:#58f;color:#fff;border-color:#58f}
 textarea{width:100%;height:9rem;font-family:ui-monospace,monospace;font-size:.85rem}
 .bar{position:fixed;left:0;right:0;bottom:0;padding:10px 16px;background:#111d;color:#fff;display:flex;gap:16px;align-items:center}
 .bar button{font:inherit;padding:.4rem .9rem;border-radius:8px;border:0;background:#fff;color:#111;cursor:pointer}
 note{display:block;opacity:.75;font-size:.9rem}
</style></head><body>
<h1>Blind listening test</h1>
<p class="sub">For each song: play the full song for context, then compare guitar A and guitar B.
Which sounds like a cleaner guitar — fewer other instruments leaking in, less of the guitar missing,
fewer watery artefacts? Then rate the better one for whether you would ship it.
You are not told which model is which. Some songs contain <em>no guitar at all</em>; if you hear a
"guitar" there, the model invented it.</p>
<div id="items"></div>
<div class="bar"><span id="progress">0 answered</span>
<button onclick="showResults()">Show results</button>
<span>paste these back into the chat</span></div>
<textarea id="out" hidden></textarea>
<script>
const ITEMS = __ITEMS__;
const answers = JSON.parse(localStorage.getItem("soundclipper_listening") || "{}");
function save(){ localStorage.setItem("soundclipper_listening", JSON.stringify(answers));
  document.getElementById("progress").textContent = Object.keys(answers).filter(k=>answers[k].better).length + " of " + ITEMS.length + " answered"; }
function pick(id, field, value, group){
  answers[id] = answers[id] || {}; answers[id][field] = value;
  document.querySelectorAll('[data-group="'+group+'"]').forEach(b=>b.setAttribute("aria-pressed", String(b.dataset.value===String(value))));
  save(); }
function note(id, value){ answers[id] = answers[id] || {}; answers[id].note = value; save(); }
function showResults(){ const box=document.getElementById("out"); box.hidden=false;
  box.value = JSON.stringify(answers, null, 1); box.select(); }
const container = document.getElementById("items");
ITEMS.forEach((item, index) => {
  const div = document.createElement("div"); div.className = "item";
  div.innerHTML = `<h2>${index+1}. ${item.title}</h2>
    <div class="row"><span>Full song</span><audio controls preload="none" src="${item.mixture}"></audio></div>
    <div class="row"><span>Guitar A</span><audio controls preload="none" src="${item.a}"></audio>
                     <span>Guitar B</span><audio controls preload="none" src="${item.b}"></audio></div>
    <div class="row choice"><span>Cleaner guitar:</span>
      <button data-group="${item.id}-better" data-value="A" onclick="pick('${item.id}','better','A','${item.id}-better')">A</button>
      <button data-group="${item.id}-better" data-value="B" onclick="pick('${item.id}','better','B','${item.id}-better')">B</button>
      <button data-group="${item.id}-better" data-value="same" onclick="pick('${item.id}','better','same','${item.id}-better')">No difference</button></div>
    <div class="row stars"><span>Would you ship the better one? 1 = unusable, 5 = ship it:</span>
      ${[1,2,3,4,5].map(n=>`<button data-group="${item.id}-usable" data-value="${n}" onclick="pick('${item.id}','usable',${n},'${item.id}-usable')">${n}</button>`).join("")}</div>
    <div class="row"><input style="flex:1;padding:.4rem" placeholder="optional note (what did you hear?)"
      oninput="note('${item.id}', this.value)" value="${(answers[item.id]&&answers[item.id].note)||""}"></div>`;
  container.appendChild(div);
});
Object.entries(answers).forEach(([id, a]) => {
  if(a.better) document.querySelectorAll('[data-group="'+id+'-better"]').forEach(b=>b.setAttribute("aria-pressed", String(b.dataset.value===a.better)));
  if(a.usable) document.querySelectorAll('[data-group="'+id+'-usable"]').forEach(b=>b.setAttribute("aria-pressed", String(b.dataset.value==String(a.usable))));
});
save();
</script></body></html>
"""
    (out / "index.html").write_text(html.replace("__ITEMS__", payload), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", default="runs/listening_test")
    parser.add_argument("--seconds", type=float, default=20.0)
    parser.add_argument("--weight", type=float, default=0.5, help="community share in the ensemble")
    parser.add_argument("--guitar-free", type=int, default=2, help="songs with no guitar, as a trap")
    parser.add_argument("--device", choices=("cpu", "cuda"))
    args = parser.parse_args()

    import soundfile as sf
    from index_moisesdb import load_moisesdb_song
    from separators import make_separator

    out = Path(args.out)
    (out / "clips").mkdir(parents=True, exist_ok=True)
    index = {r["song_id"]: r for r in (json.loads(l) for l in
                                       Path("data/moisesdb_index.jsonl").read_text().splitlines() if l.strip())}
    split = json.loads(Path("data/moisesdb_split.json").read_text())["songs"]
    scored = {}
    for path in (Path("runs/moisesdb_htdemucs_6s/tracks")).glob("*.json"):
        if path.name.startswith("probe_"):
            continue
        row = json.loads(path.read_text())
        if split.get(row["song_id"]) == "validation" and row["guitar_improvement_db"] is not None:
            scored[row["song_id"]] = row
    songs = pick_songs(scored)
    silent = [r["song_id"] for r in index.values() if r["problems"] == ["no guitar"]][:args.guitar_free]
    all_songs = songs + silent
    key = assign_slots(all_songs, MODELS)

    separators = {name: make_separator(name, args.device, args.weight) for name in MODELS}
    items = []
    for song in all_songs:
        record = index[song]
        mixture, guitar_reference = load_moisesdb_song(record)
        start = loudest_window(guitar_reference if song not in silent else mixture, args.seconds)
        n = int(args.seconds * SAMPLE_RATE)
        folder = out / "clips" / song[:8]
        folder.mkdir(parents=True, exist_ok=True)
        sf.write(folder / "mixture.wav", mixture.T[start:start + n], SAMPLE_RATE, subtype="PCM_16")
        if song not in silent:
            sf.write(folder / "real_guitar.wav", guitar_reference.T[start:start + n], SAMPLE_RATE, subtype="PCM_16")
        for slot, model in key[song].items():
            estimate = separators[model].separate_audio(mixture.T, SAMPLE_RATE)["guitar"]
            sf.write(folder / f"{slot}.wav", estimate[start:start + n], SAMPLE_RATE, subtype="PCM_16")
        row = scored.get(song)
        items.append({"id": song[:8], "title": f"{record.get('artist') or 'unknown'} — {record.get('song') or song[:8]}"
                                               + ("  (no guitar in this song)" if song in silent else ""),
                      "mixture": f"clips/{song[:8]}/mixture.wav", "a": f"clips/{song[:8]}/A.wav",
                      "b": f"clips/{song[:8]}/B.wav"})
        label = "no guitar" if song in silent else f"{row['guitar_improvement_db']:+.1f} dB"
        print(f"prepared {song[:8]} ({label})", flush=True)
    random.Random(11).shuffle(items)
    write_page(out, items)
    (out / "key.json").write_text(json.dumps({"slots": key, "weight": args.weight,
                                              "scores": {s: scored[s]["guitar_improvement_db"] for s in songs},
                                              "guitar_free": silent}, indent=2))
    print(f"\nOpen {out / 'index.html'} in your browser. The answer key is in key.json — don't peek.")


if __name__ == "__main__":
    main()
