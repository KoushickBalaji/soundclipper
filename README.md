# SoundClipper

Extract guitar and a backing track using the official pretrained **HTDemucs six-source**
model (`htdemucs_6s`). A separate preparation pipeline builds guitar/rest references
from Slakh2100-redux for a later fine-tuning experiment.

**Status: experimental research project.** The default path uses pretrained
weights, not a newly trained model. Guitar isolation quality varies by recording;
fine-tuning the pretrained model is unfinished. The repository includes offline
tests for audio reconstruction, scoring, dataset preparation, and evaluation.

## Install

Use Python 3.11. Clone the repository and create a virtual environment:

```powershell
git clone https://github.com/KoushickBalaji/soundclipper.git
cd soundclipper
py -3.11 -m venv .venv
& .venv/Scripts/python.exe -m pip install -r requirements.txt
```

On Linux/macOS, use `python3.11 -m venv .venv` and
`.venv/bin/python -m pip install -r requirements.txt`. The commands below use
Windows paths; substitute `.venv/bin/python` on Linux/macOS.

For CUDA, install matching PyTorch and torchaudio builds for your platform before
installing the requirements. CPU inference is supported but can be slow. The
first separation downloads pretrained weights and requires internet access.
No dataset is needed to separate your own WAV/FLAC recording.

Dependencies are pinned to the existing working environment; a fresh installation
on other machines still needs verification. Tests use small synthetic inputs and
do not download models or establish real-world separation quality.

## What is included

| Area | Entry points |
| --- | --- |
| Guitar and backing extraction | `pretrained_separator.py` |
| Optional RoFormer and ensemble experiments | `roformer_guitar.py`, `ensemble_separator.py`, `separators.py` |
| Dataset preparation | `prepare_slakh.py`, `build_dataset.py`, `index_moisesdb.py`, `cambridge_mt.py` |
| Separation benchmarks and paired comparisons | `evaluate_slakh.py`, `evaluate_moisesdb.py`, `compare_runs.py` |
| MIDI/chord diagnostics and listening clips | `midi_notes.py`, `chords.py`, `diagnose_guitar_buckets.py`, `make_listening_clips.py` |
| Earlier training experiment | `train.py`, `make_fake_data.py` |

Most command-line tools accept `--help`. The `.bat` files are local Windows
batch workflows that assume datasets and prior outputs already exist; start with
the single-file separation command below.

Datasets, downloaded model code/weights, and generated results are kept locally
and excluded from Git. See [THIRD_PARTY.md](THIRD_PARTY.md) for sources and terms.
An open-source reuse license has not yet been selected.

## Extract guitar

From this project folder, using the existing environment:

```powershell
& .venv/Scripts/python.exe pretrained_separator.py song.wav --out runs/my_song
```

Outputs are `mixture.wav`, `guitar.wav`, `backing.wav`, and `separation.json`.
WAV/FLAC input is read with SoundFile. Mono audio is duplicated into stereo and
resampled using Demucs's band-limited converter. Outputs use the model's 44.1 kHz
rate and FLOAT WAV encoding, preserving values above full scale without clipping.
Backing is the resampled mixture minus the estimated guitar, so the two outputs
reconstruct the input. Files are not independently volume-normalized.

Python API:

```python
from pretrained_separator import GuitarSeparator

separator = GuitarSeparator()  # Uses CUDA if available; otherwise CPU.
files = separator.separate_file("song.wav", "runs/my_song")
```

`separate_audio(audio, sample_rate)` accepts NumPy mono `(samples,)` or stereo
`(samples, channels)` audio and returns a dictionary of stereo NumPy arrays:
`mixture`, `guitar`, and `backing`. Reuse one separator for multiple songs.
Downloaded official weights are cached under `models/`. The cached legacy
checkpoint is checked against the publisher's SHA-256 prefix when loaded.

## BabySlakh baseline

```powershell
& .venv/Scripts/python.exe baseline_babyslakh.py
```

This processes all 20 original BabySlakh tracks without training. Per-song audio,
reference guitars, scores, and aggregate results are under
`runs/babyslakh_htdemucs_6s/`. Each local run creates a `REPORT.md` with results
and listening links; generated reports and audio are not included in the repo.
Scores compare the estimated guitar with its reference at the native 16 kHz rate;
stereo estimates are averaged to match mono references. The mixture's own score
provides the baseline, so positive improvement means better guitar isolation.

The first full run completed all 20 tracks. Quality was inconsistent: 10 improved,
and the mean guitar improvement was approximately -4 dB, driven partly by severe
failures. These synthesized, bandwidth-limited examples do not establish quality
on real recordings. Upsampling 16 kHz input cannot restore missing high frequencies.
The wrapper's output was verified to match the official Demucs API exactly on a
12-second sample. No model weights were updated.

## Full dataset preparation

See [DATA_PREPARATION.md](DATA_PREPARATION.md) for the validated audio contract,
progress files, restart instructions, excluded tracks, and official split handling.
The original dataset remains untouched. Preparation creates native 44.1 kHz mono
references and duplicates channels only when loading a crop for the model.

The existing `train.py` is still the earlier from-scratch experiment. Fine-tuning
the pretrained six-source model against the new manifests is the next stage;
preparing data does not start training.

## Verification

```powershell
& .venv/Scripts/python.exe -m unittest discover -v
```

Verified environment: Python 3.11, Demucs 4.1.0, PyTorch 2.11.0 with CUDA 12.8,
and an NVIDIA GeForce RTX 5070 Ti. No package upgrades were needed.

All 97 offline tests passed locally on 12 October 2026. GitHub Actions runs the
same test suite with CPU PyTorch; GPU/model inference is a separate check.

## Optional experiments

For the community RoFormer path, install `requirements-community.txt`, review
[THIRD_PARTY.md](THIRD_PARTY.md), then run `fetch_community_model.py` to fetch its
configuration, checkpoint, and source code locally. Downloads follow upstream
`main` branches, so these experiments are not fully pinned; recorded SHA-256
hashes describe downloaded files rather than verify them against a trusted digest.

`plot_spectrograms.py` additionally requires `matplotlib` and a completed local
BabySlakh baseline run. Evaluation tools require the relevant datasets and
prepared manifests. They are not part of the single-song quick start.

## Next steps

- Verify the install in a clean environment and pin model-code revisions.
- Publish a small listening demo using audio cleared for redistribution.
- Select a reuse license for original project code.
- Implement and evaluate pretrained-model fine-tuning.

Sources: [official Demucs](https://github.com/adefossez/demucs),
[Slakh utilities](https://github.com/ethman/slakh-utils).
