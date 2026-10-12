# Slakh training data preparation

## Current inputs and outputs

`prepare_slakh.py` prepares the extracted **Slakh2100-redux** release at
`data/raw/slakh2100_flac_redux`. The raw release is never changed.

- Official directories remain `train` (1,289 tracks), `validation` (270), and
  `test` (151). All 390 `omitted` tracks are excluded.
- The initial audit found no duplicate MIDI UUIDs within the 1,710 permitted
  tracks. UUID matches across splits would quarantine every member of that
  group. This check cannot identify near-duplicate performances with different
  UUIDs; the publisher's redux split already removes known duplicated songs.
- The native audio totals 118.26 hours. The maximum derived storage is 150.2 GB;
  the initial preflight found 447.9 GB free. No stereo audio is duplicated on disk.
- Two tracks (`train/Track00590`, `train/Track00749`) have no rendered guitar
  according to metadata and are expected to be skipped. All data issues are
  recorded with a reason, rather than silently assigning missing guitars to rest.
- BabySlakh is a separate 16 kHz prototype dataset. This preparer does not
  upsample it or merge it into the full training set.

## Exact audio contract

For each accepted track, `data/slakh_prepared/<official split>/<track>/` contains:

| File | Contents |
| --- | --- |
| `guitar.wav` | Sum of all rendered guitar stems; mono, native 44,100 Hz, FLOAT WAV |
| `other.wav` | Original mixture minus guitar; mono, 44,100 Hz, FLOAT WAV |
| `record.json` | Original UUID, source paths, gain provenance, integrity measurements, resume signatures |

`train.jsonl`, `validation.jsonl`, and `test.jsonl` contain accepted records only.
Manifests are written atomically at the end of a run. While a full run is active,
earlier pilot manifests can still exist; inspect the run status before consuming
the final manifests. Paths in records are absolute to this machine; if data is
moved, rerun preparation with the new roots to regenerate paths.

The original `mix.flac` is referenced, not copied. Reconstruction satisfies
`guitar + other = mixture` to float32 precision. Rest retains the tiny quantization
difference between the original mixture and the sum of original stems. It
contains **every non-guitar source combined**, including drums, bass and piano;
it is different from the pretrained six-source model's single `other` output.
When fine-tuning six-source Demucs, compare its guitar output with guitar and
the sum of its other five outputs with rest, or adapt the model deliberately.

`load_prepared_audio(record, start_frame=0, frames=-1)` returns NumPy float32:

- mixture: `(2, samples)`;
- targets: `(2 sources, 2 channels, samples)`, source order **guitar, other**.

Both channels are identical copies of the original mono waveform, with unchanged
amplitude. Cropping uses identical native sample positions in every file. There
is no resampling, augmentation, implicit padding, or independent normalization.
Use `torch.from_numpy(...)` to obtain model tensors. The loader is independent
of the existing trainer; the trainer has not yet been switched to these manifests.

## Gain and quality checks

Slakh stems already have the publisher's loudness and peak gain applied.
**Do not multiply by metadata `overall_gain` a second time.** The publisher's
[submix code](https://github.com/ethman/slakh-utils/blob/master/submixes/submixes.py)
explicitly sums delivered stems without another gain. The
[metadata documentation](https://github.com/ethman/slakh-utils#metadata) describes
the gain applied when creating the release.

Measured on the first training, validation and test track, direct summation
matched the mixture with RMS residuals of approximately 0.00006–0.00008 over the
first 30 seconds. Reapplying `overall_gain` increased the error to 0.067–0.086.

Before accepting a track, preparation checks:

1. Metadata and every rendered stem exist, with no unexpected extra audio.
2. Mixture and all rendered stems have identical length, native 44.1 kHz rate,
   and one channel. No truncation, padding, or repair conceals misalignment.
3. Every sample of every file can be decoded and is finite.
4. The sum of **all** stems matches the mixture within the combined sample
   quantization bound (one LSB per PCM source and mixture).
5. Guitar, rest, and mixture are not completely silent. Silent individual stems
   are reported in the record. Very quiet nonzero audio is retained.
6. Saved guitar plus rest reconstructs mixture within `1e-6` amplitude.

Guitar means `inst_class == "Guitar"` or zero-based MIDI programs 24–31, provided
the stem is not flagged as drums. An unrendered guitar with no audio is listed in
the record; a missing **rendered** guitar rejects the entire track. The full
release's metadata is checked strictly. BabySlakh has unreliable
`audio_rendered: false` flags for files that exist and requires its separate loader.

## Running and resuming

```powershell
& .venv/Scripts/python.exe -m unittest test_prepare_slakh -v
& .venv/Scripts/python.exe prepare_slakh.py --workers 2
```

A small pilot uses the first two complete tracks in **each** split:

```powershell
& .venv/Scripts/python.exe prepare_slakh.py --max-tracks-per-split 2 --run-dir runs/slakh_preparation/pilot
```

To resume in a hidden background process after shutdown:

```powershell
Start-Process -FilePath 'C:\Projects\soundclipper\.venv\Scripts\python.exe' -ArgumentList @('-u','C:\Projects\soundclipper\prepare_slakh.py','--workers','2') -WorkingDirectory 'C:\Projects\soundclipper' -WindowStyle Hidden -RedirectStandardOutput 'C:\Projects\soundclipper\runs\slakh_preparation\stdout.log' -RedirectStandardError 'C:\Projects\soundclipper\runs\slakh_preparation\stderr.log'
```

The operating-system lock on `data/slakh_prepared/prepare.lock` prevents concurrent
writers. Per-track records are saved after both audio files have been finalized.
Resume accepts an existing record only when the metadata fingerprint, input file
sizes/timestamps and output sizes/timestamps match. Interrupted partial files
are regenerated; unchanged complete tracks are reused. Resume signatures detect
ordinary changes but are not a cryptographic re-verification of all audio bytes.

Progress and reports are under `runs/slakh_preparation/`:

- `status.json`: current PID, completed/total, per-split accepted counts, reasons,
  elapsed time and estimated remaining time; updated after each track.
- `audit.json`: split/UUID audit and storage preflight.
- `events.jsonl`: append-only per-track outcomes across runs.
- `stdout.log`, `stderr.log`: background process messages.
- `summary.json`, `skipped.json`: final counts and all rejected/error records.

Keep the computer awake. A stale status timestamp can mean the process stopped;
check its PID before resuming. Preparation uses two CPU workers and bounded audio
blocks and does not run a model, training, or test-set scoring.

## Verification performed

Ten synthetic regression tests cover preapplied gain, waveform reconstruction,
stereo loading/cropping, official split preservation, exclusion of omitted files,
UUID leakage quarantine, missing guitar, misalignment, incorrect gain, silence,
resume recovery, drum classification, and preventing a second writer from
overwriting the active run's status. A six-track pilot passed with no errors and
exact reconstruction of all saved float32 references.
