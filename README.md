# `SolvingLinearEquations` Sample
## Introduction

Solving linear equations is a common problem in scientific computing. The `SolvingLinearEquations` is a simple program that solves linear equations using DPC++ for Intel CPU and accelerators.

This program implements a simple iterative method which is similar to Jacobi iteration method and G-S iteration method, to solve linear equations. However, the calculation may not converge currently.

## How to build for DPC++ on Linux  

   * Build the program using Make

     ```bash
     make all
     ```

   * Run the program using Make

     ```bash
     make run
     ```

* Clear the program using Make

  ```bash
  make clean
  ```

## Running the Sample

### Application Parameters

You can modify the number of linear equations by adjusting the constant parameter in `solver.cpp`. The configurable parameters include: `constexpr N = 10`.

### Example of Output
```
 ./solver
Device: Intel(R) Xeon(R) Gold 6128 CPU @ 3.40GHz
The number of equations: N 10
The linear equations:
6	1	1	1	1	1	1	1	1	1	66	
1	51	3	4	5	6	7	8	9	10	-93	
1	3	94	7	9	11	13	15	17	19	60	
1	4	7	135	13	16	19	22	25	28	-12	
1	5	9	13	174	21	25	29	33	37	0	
1	6	11	16	21	211	31	36	41	46	-23	
1	7	13	19	25	31	246	43	49	55	87	
1	8	15	22	29	36	43	279	57	64	-30	
1	9	17	25	33	41	49	57	310	73	-62	
1	10	19	28	37	46	55	64	73	339	-32	
The solution of the equations using DPC++:
11.2714	-2.04761	0.617725	-0.117698	-0.00256087	-0.114008	0.436453	-0.102494	-0.216983	-0.0811296	
Success!
```
