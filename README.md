# BMW // This is w/o scheduler branch
Beyond the Memory Wall of GPU

## 0311
Swap out communication implementation done.
APP:
swap_entry_list {gpu_address, cpu_address, size} need verification 
Swap out logic verification 
MMP:
create swap_entry_list 
swap out logic update (del will be automatically done by cudaFree from APP, add to swap_entry_list needed)

Need Total verification // i'm lossing control of flows
Concerns
1. Named Pipe mixed up // eviction_protocal vs Sendrequest due to synchronous ack 
2. Data structure soundness (update)







TODO<br>

