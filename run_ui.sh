#!/bin/sh
# Launch the HR Assistant UI from the project root so relative paths
# (data/, hr_index/, logs/) resolve correctly.
#
# --server.fileWatcherType none is REQUIRED: Streamlit's module watcher crawls
# transformers' submodules and imports accelerate.state concurrently with model
# loading, which triggers a circular-import crash ("partially initialized module
# accelerate.state"). Disabling the watcher avoids the race. (Also set in
# .streamlit/config.toml, but passing it here makes it independent of CWD.)
cd "$(dirname "$0")" || exit 1
exec .venv/bin/streamlit run src/app.py \
  --server.port "${PORT:-8501}" --server.address 0.0.0.0 --server.headless true \
  --server.fileWatcherType none
