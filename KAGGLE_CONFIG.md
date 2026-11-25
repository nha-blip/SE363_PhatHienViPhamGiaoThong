# Two-model Kaggle configuration

For two background GPU workers with per-epoch Google Drive backups, see `KAGGLE_DRIVE_TRAINING.md` and `scripts/kaggle_train_drive.py`. Drive authorization must use Kaggle Secrets, not a public notebook or Dataset.

Upload `scripts/kaggle_training_config.py` as notebook code or as a small additional private Kaggle Dataset. The script only generates configurations; it never trains, copies images, rewrites labels or changes source YAML files. The mounted example below assumes the uploaded file is at the Dataset root; include `/scripts/` if you upload the folder intact.

Run in a notebook cell, replacing only the script path:

```python
!python /kaggle/input/<code-dataset>/kaggle_training_config.py --check-pairs
```

Alternatively, paste the entire Python script into one notebook cell and run it. The script filters only Jupyter's `-f kernel-*.json` connection argument when running inside ipykernel, while still rejecting unknown training/configuration arguments. After the pasted cell, call `main(["--check-pairs"])` in another cell to include the image/label count checks. Explicit `main([...])` arguments override the kernel's command line. A direct pasted run generates configs without pair checks, unless those are explicitly requested; no training starts.

The defaults are Kaggle paths. The same code can run in Colab, but there you must first mount/extract the datasets and explicitly call `main(["--input-root", "/content/datasets", "--output-dir", "/content/configs", "--check-pairs"])`. Generated train YAMLs still use the Kaggle output project path; change that working configuration to `/content/runs` before any Colab training. The script does not mount Kaggle datasets in Colab automatically.

PyYAML is required (`pip install pyyaml` if unavailable). The script discovers the two mounted datasets by their class schemas and checks the existing `images/` and `labels/` splits. It ignores the old Windows root in TrafficLight363's source YAML and writes new runtime YAML files outside the releases:

- `/kaggle/working/configs/vehicle_dataset.yaml`
- `/kaggle/working/configs/traffic_light_dataset.yaml`
- `/kaggle/working/configs/vehicle_train.yaml`
- `/kaggle/working/configs/traffic_light_train.yaml`

If multiple versions are mounted, pass `--vehicle-root <actual-root>` and `--traffic-light-root <actual-root>`. Search is bounded to six directory levels; deeper mounts also require explicit roots. Existing generated configurations are reused only when identical; otherwise choose a new output directory.

Both baseline models use `yolov8n.pt`, 640 pixels, batch 8 and two workers. Vehicle uses GPU 0; traffic light uses GPU 1. Traffic-light hue/saturation augmentation is disabled to preserve color-state semantics. One hundred epochs is an initial experiment budget, not a promise of completion within a Kaggle session. Tune batch/epochs after smoke testing. Image caching is disabled; this does not disable all label-cache attempts in the loader.

The source TrafficLight YAML has unquoted `off`, which PyYAML's YAML 1.1 loader interprets as boolean false. The generator recognizes this exact four-color schema and writes quoted `'off'` in the new runtime config. Source YAML and labels are unchanged.

Before training, check two visible GPUs (`nvidia-smi`), pin a tested Ultralytics version and record the environment. Download the shared pretrained weights once before starting two processes to avoid simultaneous downloads into the same file. Perform integrity/hash preflight and inspect loader counts/warnings. `--check-pairs` checks paths and counts, not image readability, label syntax or full release hashes.

The train YAML files can be supplied through Ultralytics CLI `cfg=...`, with each CLI launched as a separate subprocess. Do NOT use `device=[0,1]` for either model, and do not call two blocking training commands sequentially when parallel training is intended. Output directories are distinct. `last.pt` is the interrupted-run resume checkpoint; periodic checkpoints are additionally requested every five epochs. GPU/CPU/RAM/disk budgets and output persistence must be checked before long training. No launcher is executed by this config generator.

For resume, use the appropriate model's interrupted `last.pt` with `resume=True`; do not blindly rerun the initial training YAML, which deliberately starts a fresh run. Smoke-test changes should be made only to working configurations, never dataset releases. Consult the [official train configuration and resume documentation](https://docs.ultralytics.com/modes/train/).
