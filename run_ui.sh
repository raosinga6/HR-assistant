#!/bin/sh
# Launch the HR Assistant UI from the project root so relative paths
# (data/, hr_index/, logs/) resolve correctly.
cd "$(dirname "$0")" || exit 1
exec .venv/bin/streamlit run src/app.py \
  --server.port "${PORT:-8501}" --server.address 0.0.0.0 --server.headless true
