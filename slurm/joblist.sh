sacct -u bbasseri --format=JobId,Partition,NCPUS,ReqTRES,ReqMem,Start,End,State | \
awk 'NR==1{sub("ReqTRES", "GPUs")} {gsub(/billing=/, "    "); gsub(/\+/, "     ");}1'