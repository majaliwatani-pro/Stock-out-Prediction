#!/usr/bin/env bash
# Helper instructions to download Rossmann or Favorita from Kaggle.
# Kaggle CLI requires API key. See https://github.com/Kaggle/kaggle-api
#
# Steps:
# 1) Install kaggle: pip install kaggle
# 2) Place your kaggle.json in ~/.kaggle/kaggle.json (create a token from kaggle.com)
# 3) Run for Rossmann:
#    kaggle competitions download -c rossmann-store-sales -p data/
# 4) Unzip and inspect files, then adapt src/data_prep.py to extract required columns.
#
echo "See instructions in this file. Kaggle downloads are not automatic in this demo."
