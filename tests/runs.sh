#!/bin/bash
for i in 100 500 700 1000
do
    python3 filaments_strc_param/tests/test_mapping.py >> ./filaments_strc_param/tests/test_mapping.log --dim $i --lag 10 --method sequence
    python3 filaments_strc_param/tests/test_mapping.py >> ./filaments_strc_param/tests/test_mapping.log --dim $i --lag 10 --method parallel
done