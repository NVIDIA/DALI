from nvidia.dali.backend_impl import *

# Note: If we every need to add more complex functionality
# for importing the dali c++ extensions, we can do it here

initialized = False
if not initialized:
    Init(OpSpec("CPUAllocator"), OpSpec("PinnedCPUAllocator"), OpSpec("GPUAllocator"))
    initialized = True
