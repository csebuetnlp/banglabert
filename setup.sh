#!/bin/bash

git clone https://github.com/huggingface/transformers.git
cd transformers/
git checkout 7a26307e3186926373cf9129248c209ab869148b
pip install --upgrade ./
cd ../

pip install --upgrade -r requirements.txt