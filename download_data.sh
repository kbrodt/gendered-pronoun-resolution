#!/bin/bash

mkdir data
cd data
wget https://raw.githubusercontent.com/google-research-datasets/gap-coreference/master/gap-test.tsv
wget https://raw.githubusercontent.com/google-research-datasets/gap-coreference/master/gap-development.tsv
wget https://raw.githubusercontent.com/google-research-datasets/gap-coreference/master/gap-validation.tsv
