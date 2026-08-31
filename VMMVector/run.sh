#!/bin/bash
set -e
set -o pipefail

make debug
./main 2000000 2 |& tee out.txt

