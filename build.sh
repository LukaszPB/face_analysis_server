#!/usr/bin/env bash
set -o errexit  

# Aktualizacja pip/setuptools/wheel, żeby uniknąć problemów z kompilacją
pip install --upgrade pip setuptools wheel

# Instalacja systemowych pakietów dla dlib
apt-get update
apt-get install -y cmake g++ make

# Instalacja pythonowych paczek
pip install -r requirements.txt