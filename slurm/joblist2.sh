sacct -u bbasseri --format=JobId%-8,JobName%16,Partition,ReqCPUS%7,ReqTRES,ReqMem,Start,End,State%16 | awk '
NR==1 {
    # Rename header
    sub("ReqTRES", "GPUs")
    # Apply original billing substitution (if needed in header)
    gsub(/billing=/, "        ")
    print $0
    next
}
{
    # Apply original billing substitution globally first
    gsub(/billing=/, "        ")

    # Check the 5th field (ReqTRES) for the GPU pattern and replace it with the digit if found
    # The sub function returns 1 on success, 0 otherwise
    if (!sub(/.*gres\/gpu=([0-9]).*/, "\\1", $5)) {
        # If the pattern was not found and sub returned 0, set the field to 0
        $5 = "0"
    }
    print $0
}'