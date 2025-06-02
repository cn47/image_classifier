#!/bin/bash
set -e

COMMAND=$1          # up / down
FLAVOR=${2:-gpu}    # cpu or gpu (default: gpu)
PYTHON_VER=3.13.3

# set container working- dir
HOST_PATH=$(cd .. && pwd) # 現在のディレクトリの1つ上を取得
RELATIVE_PATH=${HOST_PATH#${HOME}/} # hostの$HOMEをベースに相対パスを作る(/home/{host}/以降だけ取り出す)
WORKING_DIR="/home/jupyter/${RELATIVE_PATH}"

if [ "$FLAVOR" != "gpu" ] && [ "$FLAVOR" != "cpu" ]; then
  echo "❌ Error: Unknown flavor $FLAVOR. Use 'gpu' or 'cpu'."
  exit 1
fi

export FLAVOR=$FLAVOR
export IMAGE_NAME="jupyterlab-$FLAVOR:$PYTHON_VER"
export RUNTIME=$([ "$FLAVOR" = "gpu" ] && echo "nvidia" || echo "")
export WORKING_DIR=$WORKING_DIR

main() {
  case "${COMMAND}" in
    down ) docker compose down ;;
    up ) docker compose down; docker compose up --build -d ;;
    * ) echo "❌ Invalid command: $COMMAND. Use 'up' or 'down'." ;;
  esac
}

main
