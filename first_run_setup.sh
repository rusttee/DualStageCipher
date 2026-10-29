#!/usr/bin/env bash
set -e
cd "$(dirname "$0")"
python3 -m venv .venv
. .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
python prepare_offline.py --cache-wheels
python check_offline_ready.py
