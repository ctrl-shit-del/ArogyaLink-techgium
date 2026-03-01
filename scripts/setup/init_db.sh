#!/bin/bash
# Run Alembic migrations + seed patients
set -e
alembic upgrade head
python -m scripts.setup.seed_data
