#!/bin/bash

# 这里是你要执行的命令
command1="python eva_model.py -p1 /data/0shared/zhangshanwei/ECG_founder/ECGFounder-master/PatchEncoder/picture/different_layout/5fold/layout_0 -p2 /data/0shared/zhangshanwei/ECG_founder/ECGFounder-master/PatchEncoder/table/df/5fold.xlsx -l 0 -v --verbose"
command2="python eva_model.py -p1 /data/0shared/zhangshanwei/ECG_founder/ECGFounder-master/PatchEncoder/picture/different_layout/5fold/layout_1 -p2 /data/0shared/zhangshanwei/ECG_founder/ECGFounder-master/PatchEncoder/table/df/5fold.xlsx -l 1 -v --verbose"
command3="python eva_model.py -p1 /data/0shared/zhangshanwei/ECG_founder/ECGFounder-master/PatchEncoder/picture/different_layout/5fold/layout_2 -p2 /data/0shared/zhangshanwei/ECG_founder/ECGFounder-master/PatchEncoder/table/df/5fold.xlsx -l 2 -v --verbose"
command4="python eva_model.py -p1 /data/0shared/zhangshanwei/ECG_founder/ECGFounder-master/PatchEncoder/picture/different_layout/5fold/layout_3 -p2 /data/0shared/zhangshanwei/ECG_founder/ECGFounder-master/PatchEncoder/table/df/5fold.xlsx -l 3 -v --verbose"
command5="python eva_model.py -p1 /data/0shared/zhangshanwei/ECG_founder/ECGFounder-master/PatchEncoder/picture/different_layout/5fold/layout_4 -p2 /data/0shared/zhangshanwei/ECG_founder/ECGFounder-master/PatchEncoder/table/df/5fold.xlsx -l 4 -v --verbose"
command6="python eva_model.py -p1 /data/0shared/zhangshanwei/ECG_founder/ECGFounder-master/PatchEncoder/picture/different_layout/5fold/layout_5 -p2 /data/0shared/zhangshanwei/ECG_founder/ECGFounder-master/PatchEncoder/table/df/5fold.xlsx -l 5 -v --verbose"
command7="python eva_model.py -p1 /data/0shared/zhangshanwei/ECG_founder/ECGFounder-master/PatchEncoder/picture/different_layout/5fold/layout_6 -p2 /data/0shared/zhangshanwei/ECG_founder/ECGFounder-master/PatchEncoder/table/df/5fold.xlsx -l 6 -v --verbose"
# 执行命令
$command1
$command2
$command3
$command4
$command5
$command6
$command7
# 执行完毕

