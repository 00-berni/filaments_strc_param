#!/bin/bash
dimensions=(100 500 700 1000 2000)
maxlags=(10 100 200 400 500)
indexes=(0 1 2 3 4)

for i in ${indexes[0]}
do
    echo "===================================="
    echo "START PROCESS DIM ${dimensions[$i]}"
    current_time="`date "+%H:%M:%S"`"
    echo "Start Time: $current_time"
    echo "> start sequencial"
    python3 -m filaments_strc_param.tests.test_speed >> ./filaments_strc_param/tests/test_speed.log --dim ${dimensions[$i]} --maxlag ${maxlags[$i]} --method sequence
    echo "> end sequencial"
    echo "> start parallel"
    python3 -m filaments_strc_param.tests.test_speed >> ./filaments_strc_param/tests/test_speed.log --dim ${dimensions[$i]} --maxlag ${maxlags[$i]} --method parallel
    echo "> end parallel"
    current_time="`date "+%H:%M:%S"`"
    echo "End Time: $current_time"
    echo "END PROCESS"
    echo "===================================="
    echo ""
done
