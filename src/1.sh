#!/bin/bash

# 
command1="python eva_model.py -p1 ../eva/ex1/picture/AUC_5fold_layout_0 -p2 ../eva/ex1/table/sub_5fold.xlsx -l 0 -v --verbose"
command2="python eva_model.py -p1 ../eva/ex1/picture/AUC_5fold_layout_1 -p2 ../eva/ex1/table/sub_5fold.xlsx -l 1 -v --verbose"
command3="python eva_model.py -p1 ../eva/ex1/picture/AUC_5fold_layout_2 -p2 ../eva/ex1/table/sub_5fold.xlsx -l 2 -v --verbose"
command4="python eva_model.py -p1 ../eva/ex1/picture/AUC_5fold_layout_3 -p2 ../eva/ex1/table/sub_5fold.xlsx -l 3 -v --verbose"
command5="python eva_model.py -p1 ../eva/ex1/picture/AUC_5fold_layout_4 -p2 ../eva/ex1/table/sub_5fold.xlsx -l 4 -v --verbose"

# 
$command1
$command2
$command3
$command4
$command5


