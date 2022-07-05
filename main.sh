#!/bin/bash
for i in {0..29}; do
	for VARIABLE in "cc" "cmb" "pm1" "sct" "t481" "tcon" "vda" "alu2" "alu4" "cm85a" "cm151a" "cm162a" "cu" "x2"; do
		./${VARIABLE} ${VARIABLE} exe_$((i+1)) $i
	done
done



