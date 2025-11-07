#!/usr/bin/env bash
set -e
# Providing the required environment variables
APP_REPO_URL="${APP_REPO_URL:-https://github.com/yugesharma/mlhf.git}"
APP_DIR="${APP_DIR:-mlhf}"
VENV_DIR="${VENV_DIR:-$APP_DIR/venv}"
APP_PORT="${APP_PORT:-7860}"
APP_ENTRY="${APP_ENTRY:-app.py}"
APP_HOST="${APP_HOST:-$(hostname -f 2>/dev/null || hostname)}"
LOCK_FILE="/tmp/deploying"

#Lock mechanism to avoid duplication of deployment process
if [ -f "$LOCKFILE" ]; then
  echo "Deployment already in progress, exiting."
  exit 0
fi

touch "$LOCKFILE"

trap 'rm -f "$LOCKFILE" ' EXIT

#Updating packages and installing Python 
sudo apt-get update -y && sudo apt-get install -y --no-install-recommends git python3-venv python3-pip

#Looking for existing file directory and removing it if it exists. Cloning the repo
if [ -d "$APP_DIR/.git" ]; then
  git -C "$APP_DIR" pull --rebase || true
else
  rm -rf "$APP_DIR"
  git clone --depth 1 "$APP_REPO_URL" "$APP_DIR"
fi

#Creating a new virtual environment and installing dependencies using requirements.txt
if [ ! -x "$VENV_DIR/bin/python" ]; then python3 -m venv "$VENV_DIR"; fi
if [ -x "$VENV_DIR/bin/pip" ]; then "$VENV_DIR/bin/pip" install --upgrade pip; fi

if [ -f "$APP_DIR/requirements.txt" ]; then
  "$VENV_DIR/bin/pip" install -r "$APP_DIR/requirements.txt"
fi

#kill any previous instance of the app running
pkill -f "$VENV_DIR/bin/python $APP_ENTRY" || true


cd "$APP_DIR"
# Pass the required tokens and finally execute running the app.py uninterupted even if the shell closes with logging
HF_TOKEN="$HF_TOKEN" SLACK_WEBHOOK="$SLACK_WEBHOOK" nohup "$HOME/mlhf/venv/bin/python" "$APP_ENTRY" >> log.txt 2>&1 &

#Printing out the hosted app URL for verification that the script is complete
echo "http://$APP_HOST:$APP_PORT"

