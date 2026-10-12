# External models and datasets

SoundClipper provides orchestration, audio handling, dataset preparation, and
evaluation around external models. The pretrained models were not trained by
this project. Downloaded source code, weights, datasets, and generated audio are
excluded from this repository.

- **HTDemucs six-source model**: loaded through
  [Demucs](https://github.com/adefossez/demucs). Consult the publisher's license
  and model terms before redistribution or commercial use.
- **Community guitar RoFormer**: the optional fetch script downloads weights
  and configuration from
  [becruily/mel-band-roformer-guitar](https://huggingface.co/becruily/mel-band-roformer-guitar)
  and model code from
  [Music-Source-Separation-Training](https://github.com/ZFTurbo/Music-Source-Separation-Training).
  The existing experiment did not establish permission to redistribute these
  files; they remain local. Review current upstream terms before using them.
  Its loader uses legacy pickle checkpoints and YAML Python tags; use only
  trusted publisher files, never untrusted checkpoints or configurations.
- **Slakh / BabySlakh**: obtain data separately through the
  [Slakh project](https://www.slakh.com/) and follow its dataset terms.
- **MoisesDB**: obtain data separately through
  [MoisesDB](https://github.com/moises-ai/moises-db) and follow its dataset terms.
- **Cambridge multitracks**: obtain data separately through the
  [Cambridge Music Technology library](https://www.cambridge-mt.com/ms/mtk/)
  and respect the permissions attached to each recording.

Only process recordings you have permission to use. Dataset downloads can be
large; consult each script's help before starting a full preparation run.

No project-wide reuse license has been selected yet. This repository does not
grant additional rights to external code, models, datasets, or recordings.
