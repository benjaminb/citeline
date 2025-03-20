sacct -u bbasseri --format=JobId,Partition,NCPUS,ReqTRES,ReqMem,Start,End,State | \
awk 'NR==1{sub("ReqTRES", "GPU")} {gsub(/billing=[0-9]+\+/, substr($4, 9, length($4)-9))}1'