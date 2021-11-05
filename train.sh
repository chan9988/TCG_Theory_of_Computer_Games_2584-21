#!/bin/bash

./2584 --total=0 --play="init save=weights.bin"
for i in {1..100}; do
	./2584 --total=200000 --block=1000 --limit=1000 --play="load=weights.bin save=weights.bin alpha=0.0025" | tee -a train.log
	./2584 --total=1000 --play="load=weights.bin alpha=0" --save="stat.txt"
	tar zcvf weights.$(date +%Y%m%d-%H%M%S).tar.gz weights.bin train.log stat.txt
done
