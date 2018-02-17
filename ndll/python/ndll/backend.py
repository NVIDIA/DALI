from ndll.ndll_backend import *

# Note: If we every need to add more complex functionality
# for importing the ndll c++ extensions, we can do it here

initialized = False
if not initialized:
    Init(OpSpec("CPUAllocator"), OpSpec("PinnedCPUAllocator"), OpSpec("GPUAllocator"))
    initialized = True
