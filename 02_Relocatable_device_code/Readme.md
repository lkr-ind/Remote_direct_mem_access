# RDC (Relocatable Device Code) in CUDA Compilation and Linking

RDC (Relocatable Device Code) is an important feature in CUDA that enables separate compilation and linking of device code. It allows device functions (`__device__` and `__global__`) to be defined in one compilation unit (e.g., a .cu file) and used in another. Without RDC, all device functions must be known at compile-time within a single translation unit.

There are 3 files in this directory that demonstrate how to use RDC in CUDA compilation and linking:

1. [device_functions.cuh](./device_functions.cuh): A header file declaring a device function `multiply`.

2. [device_functions.cu](./device_functions.cu): A CUDA source file defining the device function `multiply`.

3. [main.cu](./main.cu): A CUDA source file that uses the device function `multiply`.

## Compiling code without RDC

At first, let us try to compile the code without RDC. To do this, we will compile the `device_functions.cu` and `main.cu` files separately and then link them together.

```bash
nvcc -c device_functions.cu -o device_functions.o
```

This will compile fine since the `device_functions.cu` file contains the definition of the `multiply` function. Next, we will compile the `main.cu` file.

```bash
nvcc -c main.cu -o main.o
```

This will fail with an error message similar to the following:

```bash
ptxas fatal   : Unresolved extern function '_Z8multiplyff'
```

As can be seen from the error message, the compiler is unable to find the definition of the `multiply` function. This is where `RDC` flag comes into play and allows us to define `__device__` functions in a separate compilation unit and use in another.

## Compiling code with RDC

To compile the code with RDC, we need to add the `-rdc=true` flag to the `nvcc` command. Let us compile the `device_functions.cu` file with the RDC flag.

```bash
nvcc -rdc=true -c device_functions.cu -o device_functions.o
nvcc -rdc=true -c main.cu -o main.o
nvcc -rdc=true device_functions.o main.o -o my_cuda_program
```
