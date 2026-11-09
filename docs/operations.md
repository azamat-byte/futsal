# Operations and recovery

## Jobs

The dashboard starts a separate worker with hidden Windows process creation. SQLite stores state, PID, progress and the latest attempt path. Refresh progress explicitly. Cancel requests are checked between sampled frames. Model loading and initial source hashing may occur before the first progress update.

Restarting a failed/cancelled/interrupted job creates `attempt-0002`, etc. It reprocesses the original configured interval from its beginning. Previous output remains available. This avoids unsupported tracker-state splicing.

An OS kill can leave `running` stale. Check the PID before declaring an interruption. `futsal.jobs.update_job(root,id,state='interrupted')` permits recovery. Logs are in `runs/<job>/worker.log` and the attempt directory.

## Full videos

Select Process until end in the UI or use CLI `--full --background`. Use one GPU worker at a time. A 4 GB GPU and long 1080p sources can take substantial time. Benchmark clips first. At 2 samples/s, physical dynamics and event timing are unsuitable for fine-grained analysis; use 10 samples/s for movement/event development. Low sampling does not make unavailable ground truth available.

Decode remains streaming, and tracks are partitioned. Interactive inspection loads a bounded source-time window and caps display rows. CSV/Parquet exports stream partition batches and retain all observations. Whole-run analytical derivation can still require substantial tabular memory when valid metric observations exist; profile it before large high-cadence multi-hour deployments.

## Calibration

In Calibration, click visible landmarks, provide actual court coordinates, and designate held-out points. Save a reviewed anchor file. Re-run the source interval with that file. Short-term propagation expires after two seconds by default. This conservative expiry is intentional: persistent panning calibration needs additional anchors or a trained futsal landmark detector.

## Reproducibility

Keep manifests, environment reports, data-review history and source/model hashes. Do not overwrite raw assets. Never export API keys. The dashboard binds to localhost. No cloud upload occurs in the standard pipeline.
