#!/bin/bash
set -e
set -o pipefail

make debug
./main 1000000 4 6 #|& tee out_device.txt

