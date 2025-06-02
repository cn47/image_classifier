#!/bin/bash
set -e

PYTHON_VER=3.13.3
FLAVOR=${1:-gpu}    # cpu or gpu (default: gpu)

if [ "$FLAVOR" != "gpu" ] && [ "$FLAVOR" != "cpu" ]; then
  echo "❌ Error: Unknown flavor $FLAVOR. Use 'gpu' or 'cpu'."
  exit 1
fi

main() {
  echo "Building $FLAVOR image with Python $PYTHON_VER..."
  docker build \
    --build-arg PYTHON_VER=$PYTHON_VER \
    -t cn47/jupyterlab-$FLAVOR:$PYTHON_VER \
    -f Dockerfile.$FLAVOR.base .

  echo "Pushing image to Docker Hub..."
  docker push cn47/jupyterlab-$FLAVOR:$PYTHON_VER
}

main
