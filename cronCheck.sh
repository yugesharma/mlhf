#!/usr/bin/env bash
set -Eeuo pipefail
source "$HOME/cs553/env.sh" #Providing the file containg HuggingFace key. #Create file env.sh and add the token to it with export command

#Setting up dedicated enviroment variables for cron job for paths and keys
HOST="paffenroth-23.dyn.wpi.edu"
PORT="22001"
FAIL_USER="student-admin"
LOG_FILE="$HOME/cs553/autobot.log"
DEFAULT_STUDENT_KEY="$HOME/.ssh/student-admin_key"
GOOD_PRIV_KEY="$HOME/.ssh/no_phrase_key"
GOOD_PUB_KEY="$HOME/.ssh/no_phrase_key.pub"
GH_TOKEN="${GH_TOKEN:-}"
HF_TOKEN="${HF_TOKEN:-}"
SLACK_WEBHOOK="${SLACK_WEBHOOK:-}"
Pinecone="${Pinecone:-}"

APP_REPO_URL="${APP_REPO_URL:-https://github.com/yugesharma/mlhf.git}"
APP_DIR="${APP_DIR:-mlhf}"
VENV_DIR="${VENV_DIR:-$APP_DIR/venv}"
APP_PORT="${APP_PORT:-7860}"

SLEEP="${SLEEP:-180}"
JITTER_MAX="${JITTER_MAX:-60}"

# checks login using our key and passes commands to execute on VM
ssh_good() {
 ssh -q -o BatchMode=yes -o StrictHostKeyChecking=no -o ConnectTimeout=5 -p "$PORT" -i "$GOOD_PRIV_KEY" "$FAIL_USER@$HOST" "$@"
}
# checks login using default key 
ssh_default() {
 ssh -q -o BatchMode=yes -o StrictHostKeyChecking=no -o ConnectTimeout=5  -p "$PORT" -i "$DEFAULT_STUDENT_KEY" "$FAIL_USER@$HOST" "$@"
}
# SCP function for sending deploy.sh to VM for deployment process once the default key is replaced with our key
scp_good() {
 scp -q -P "$PORT" -i "$GOOD_PRIV_KEY" -o StrictHostKeyChecking=no "$1" "$FAIL_USER@$HOST:$2"
}
# SCP function for sending our key to the VM to replace the default key
scp_default() {
 scp -q -P "$PORT" -i "$DEFAULT_STUDENT_KEY" -o StrictHostKeyChecking=no "$1" "$FAIL_USER@$HOST:$2"
}

# Configuration for removing old hosts, transfering our key, changing permissions, and replacing the default key with our key
install_good_pubkey_via_default() {
 echo "$(date -Is) revert: installing good pubkey" >> "$LOG_FILE"
 ssh-keygen -R "[paffenroth-23.dyn.wpi.edu]:22001"
 scp_default "$GOOD_PUB_KEY" "/tmp/.goodkey.pub"
 ssh_default "mkdir -p ~/.ssh && chmod 700 ~/.ssh && cat /tmp/.goodkey.pub > ~/.ssh/authorized_keys && rm /tmp/.goodkey.pub && chmod 600 ~/.ssh/authorized_keys"
}

#Handles initial setup for deploy.sh, script for deploying. returns 0 if deployment in process. Else, it transfers the script, logs in for HuggingFace
# passes the other necessary environment variables which are added to the VM and then executes the deployment script
run_remote_bootstrap() {
  if ssh_good "[ -f /tmp/deploying ]"; then
    echo "$(date -Is) chill: deployment in progress" >> "$LOG_FILE"
    return 0
  fi
  scp_good "$HOME/cs553/deploy.sh" "/tmp/deploy.sh"
  ssh_good "sudo pip install -U 'huggingface_hub[cli]'"
  ssh_good "echo HF_TOKEN=$HF_TOKEN"
  ssh_good "huggingface-cli login --token=$HF_TOKEN"
  ssh_good "GH_TOKEN='$GH_TOKEN' APP_REPO_URL='$APP_REPO_URL' APP_DIR='mlhf' VENV_DIR='mlhf/venv' APP_PORT='$APP_PORT' APP_HOST='$HOST' HF_TOKEN='$HF_TOKEN' SLACK_WEBHOOK='$SLACK_WEBHOOK' bash /tmp/deploy.sh"
}


echo "START CRON"

#Flow which first checks if our key is working. Then checks if the product is up. If product is not up, the deploy script is triggered
if ssh_good exit; then
  echo "$(date -Is) ok: good key works">> "$LOG_FILE"
  if ! curl -fsS "http://paffenroth-23.dyn.wpi.edu:8001/" >/dev/null; then 
 echo "$(date -Is) ok: APP HEALTH=FAIL">> "$LOG_FILE"
  run_remote_bootstrap
else
 echo "$(date -Is) ok: APP HEALTH=OK">> "$LOG_FILE"
fi
#if our key fails login, tries the default key. Then transfers our key to replace the default key and runs the deploy script
elif ssh_default exit >> "$LOG_FILE"; then
  install_good_pubkey_via_default
  if ssh_good exit >> "$LOG_FILE"; then
    echo "$(date -Is) takeover ok: good key now accepted">> "$LOG_FILE"
    run_remote_bootstrap
  else
    echo "$(date -Is) warning: good key still not accepted">> "$LOG_FILE"
  fi
#else statement to handle any other cases of failure of logging in, which are usually due to network
else
  echo "$(date -Is) network/ssh issue; backoff">> "$LOG_FILE"
fi


