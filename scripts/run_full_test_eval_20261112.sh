#!/bin/bash
cd /d/se363-bigdata
PY=C:/Users/NHA/miniconda3/envs/my_env/python.exe
run() { # task imgsz [extra]
  echo "START $1 $2 $3 $(date +%T)" >> runs/project363/test_full_progress.log
  $PY scripts/analyze_vehicle_val_errors.py --execute --split test --task $1 --imgsz $2 $3 > runs/project363/test_full_$1_$2.log 2>&1
  echo "END $1 $2 exit=$? $(date +%T)" >> runs/project363/test_full_progress.log
}
run light 640
run light_ft960 960
run light_ft960 640
run light_v8s 960
run light_v8s 640
run vehicle_local_best 640 --skip-official-val
run vehicle 960
run vehicle_local49 960
run vehicle_v8s 960
run vehicle_v8s 640 --skip-official-val
echo ALLDONE >> runs/project363/test_full_progress.log
