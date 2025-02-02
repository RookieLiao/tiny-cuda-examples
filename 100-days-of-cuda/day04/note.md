Ideally, reducing among wrap is fastest. However, we need local registers to store inputs as each thread needs to store K/32*pack_size*sizeof(float) bytes.

Q: Why need to store `K/32*pack_size*sizeof(float)` bytes?
A: the GPU compiler (PTXAS) does register allocation at compile time. It decides how many registers a kernel needs at peak usage. You don’t manually free them in code like you might do with a CPU stack variable.

Peak usage: If the kernel’s code requires, say, 48 registers at its most demanding moment (e.g., right after loading multiple values), then 48 registers are allocated for that thread for the entire kernel lifetime.
If you ask the compiler to let each thread process too large a chunk (thus needing more registers than are physically available for high occupancy), the compiler will “spill” variables to shared or local memory. This is slower.
