#!/bin/bash

set -e

GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${BLUE}[INFO]${NC} Initialization started at $(date)"
echo -e "${BLUE}[INFO]${NC} Working Directory: $(pwd)"
echo -e "${BLUE}[INFO]${NC} Python Version: $(python --version)"

if [ ! -f "entry_point.py" ]; then
    echo -e "${RED}ERROR:${NC} entry_point.py not found in /app"
    exit 1
fi

echo -e "${BLUE}[INFO]${NC} Launching MindSpore Physics Engine..."
echo "----------------------------------------------------------------"

python entry_point.py "$@"

echo "----------------------------------------------------------------"
echo -e "${GREEN}[SUCCESS]${NC} Job completed successfully at $(date)"