#!/bin/bash
# Filename: setup.sh
# Description: This file sets up all required
# dependencies and the environment needed for
# the project to run.
# 
# 2021.09.22

# configuring paths and installing packages that make cuda work
apt-get update
apt-get install -y libsm6 libxext6 libxrender-dev
# installing python dependencies
pip3 install -r requirements.txt
# getting absolute path of parent directory
COVID_PATH=$( cd .. && pwd )
# /usr/local/lib/python3.6/dist-packages/vis/visualization/saliency.py
# setting python paths
echo "export PYTHONPATH="${PYTHONPATH}:${COVID_PATH}"" >> ~/.bashrc
# reloading bashrc
exec bash
