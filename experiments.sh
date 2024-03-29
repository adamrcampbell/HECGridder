#!/bin/bash

# Gridding configurables
KERNEL_TEX=128
KERNEL_RES=512
GRID_DIMENSION=18000
KERNEL_MIN_HALF=4
KERNEL_MAX_HALF=44
MAX_W=7083.386132813
NUM_W_PLANES=339
CELL_SIZE_RADS=0.00000639708380288949
FRAG_TYPE=2 # 1 = radial, 2 = reflect

# Available datasets
EL30_56=datasets/el30-56.txt
EL56_82=datasets/el56-82.txt
EL82_70=datasets/el82-70.txt 

# ./hecgridder_experiments $BIG $BIG $BIG 2 32 1 339 $CS6 7083.386050 44.0 1 $EL8270 3 3 $EL8270R $EL8270I
./dist/Debug/GNU-Linux/hecgridder $EL82_70 $KERNEL_TEX $KERNEL_RES $GRID_DIMENSION \
	$FRAG_TYPE $KERNEL_MIN_HALF $KERNEL_MAX_HALF $MAX_W $NUM_W_PLANES $CELL_SIZE_RADS