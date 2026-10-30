#!/bin/bash
# In WSL: put the stand-in's session script where 2-open-rtool.bat expects Barkla's.
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
mkdir -p /run/sshd /root/.ssh "/root/Research tool/scripts" /mnt/scratch/users/root/rtool/ui
tr -d '\r' < wsl_key.pub > /root/.ssh/authorized_keys
chmod 700 /root/.ssh
chmod 600 /root/.ssh/authorized_keys
tr -d '\r' < "$here/session.sh" > "/root/Research tool/scripts/rtool_ui_session.sh"
