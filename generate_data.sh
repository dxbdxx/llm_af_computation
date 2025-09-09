#!/bin/bash

for i in {6..25}
do
  python -m src.data.generate_apxs --name "train-$i" --min_args $i --max_args $i --num 3000 --max_processes 16
done

for i in {6..25}
do
  python -m src.data.generate_apxs --name "test-$i" --min_args $i --max_args $i --num 500 --max_processes 16
done

for i in {6..25}
do
  python -m src.data.apxs_to_afs --name "train-$i"
done

for i in {6..25}
do
  python -m src.data.apxs_to_afs --name "test-$i"
done

python -m src.data.afs_to_prompt.generate_prompt

python -m src.data.afs_to_prompt.load_txt --data_num 3000

python -m src.data.afs_to_prompt.load_txt --dataset "test" --data_num 100