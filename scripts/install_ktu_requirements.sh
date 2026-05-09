#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

python -m pip install --user --upgrade pip
python -m pip install --user -r requirements.txt

echo
echo "If /proxy/8000 returns 404, restart the Jupyter server after this install."
echo "The project will then be available at:"
echo "https://ai-notebook.ktu.edu/user/jokgri@ktu.lt/proxy/8000/"
