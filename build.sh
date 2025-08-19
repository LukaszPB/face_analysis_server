#!/usr/bin/env bash
# Zatrzymaj build jeśli coś pójdzie źle
set -o errexit  

# Render Free plan nie ma build-essential, więc doinstaluj
apt-get update
apt-get install -y cmake g++ make

# Instalacja zależności Pythona
pip install -r requirements.txt