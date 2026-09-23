# Tests

## Sep 1
- VMM supports only pinned (not managed) memory alloc
- backing device can be `DEVICE` or `NUMA_HOST` -> use numa id of cpu associated with gpu0
- also need to set access perm accordingly

Resulting times are in [out_device.txt](./out_device.txt) and [out_host_numa.txt](./out_host_numa.txt).