# Best Configuration for RotatE-VLP
#
bash run.sh train RotatE FB15k-237 0 0 1024 256 1000 11.0 1.0 0.0005 100000 16 -1 0.5 0.5 5 3.0 1.7 13 40000 -de
bash run.sh train RotatE wn18rr 0 0 512 1024 500 4.0 0.5 0.00005 80000 8 1 0.5 5.0 8 0.5 1.1 10 30000 -de
#
# Best Configuration for ComplEx-VLP
#
bash run.sh train ComplEx FB15k-237 0 0 1024 256 2000 200.0 1.0 0.001 100000 16 2 0.5 0.5 8 2.0 1.4 15 50000 -de -dr -r 0.005
bash run.sh train ComplEx wn18rr 0 0 512 1024 500 200.0 1.0 0.001 80000 8 1 0.7 0.0 5 1.5 0.1 12 30000 -de -dr -r 0.01
#
# Best Configuration for DistMult
# 
bash run.sh train DistMult FB15k-237 0 0 1024 256 2000 200.0 1.0 0.002 100000 16 -1 0.5 0.0 3 1.0 1.6 13 30000 -r 0.005
bash run.sh train DistMult wn18rr 0 0 512 1024 1000 200.0 1.0 0.0005 80000 8 1 0.1 0.05 8 1.0 0.3 8 30000 -r 0.01
#
