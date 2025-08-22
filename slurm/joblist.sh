sacct -u bbasseri --parsable2 \
  --format=JobID,JobName,Partition,ReqCPUS,ReqTRES,ReqMem,Start,End,State \
| awk -F'|' 'BEGIN{OFS="|"}
  NR==1 { print "JobID","JobName","Partition","CPUs","GPUs","ReqMem","Start","End","State"; next }
  {
    gpu=0; if (match($5,/gres\/gpu[=:]([0-9]+)/,a)) gpu=a[1];
    print $1,$2,$3,$4,gpu,$6,$7,$8,$9
  }
' | column -t -s '|'
