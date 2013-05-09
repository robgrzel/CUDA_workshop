/* Small CUDA exercise to try to improve efficiency by using two
   separate streams to set up a staged copying and execution (when
   instead of one large copy, followed by one large kernel
   computation, one does it in many small chunks, with copying and
   computing done in parallel in two streams, after first chunk was
   copied to the device).

   For the purpose of this exercise, ignore the copying of the
   results from the device to host at the end; only do the staged copy
   and execute for the copying of the initial data to the device + the
   kernel.

   We use cudaMallocHost to allocate arrays on host in pinned memory,
   which is both results in faster copying to/from GPU (compared to malloc),
   and also a CUDA requirement for copying running concurrently with a kernel.

   Make sure that the "Result:" value printed by the code is (almost)
   identical in both original and modified versions of the code. If
   not, you have a bug!

Hints: You will have to use the following CUDA functions:

 - cudaStreamCreate
 - cudaMemcpyAsync
 - cudaStreamDestroy
 - cudaDeviceSynchronize

* You will have to set up a for loop for multiple chunks copying and
  kernel execution;

* Number of chunks should be a variable (or macro parameter); for
  simplicity, make NMAX dividable by the number of chunks;

* In cudaMemcpyAsync the first two arguments should be "&d_A[ind],
  &h_A[ind]", not "d_A, h_A", ind being the starting index for the
  current chunk to copy;

* You'll have to pass two more arguments to the kernel - 
  ind and number of threads per chunk;

* Nblocks will be different - shoul be computed per chunk.

* You'll have to modify the kernel slightly;

At the end, you should get the timings (based on 10 runs, NMAX=1000000,
BLOCK_SIZE=128) similar to this:

NCHUNKS     t, ms
  -         2.76    - the original (non-staged) version of the code
  1         2.72    - result is similar to the non-staged code
  2         2.08    - even with only 2 chunks, we already see 33% speedup
  4         1.79
  5         1.75
 10         1.72    - seems to be the best timing, 60% faster than the original code
 20         1.87    - as NCHUNKS increases, the results get worse. Why?
100         3.53    - for too many NCHUNKS results can get even worse than in non-staged code

To compile:

nvcc -arch=sm_20 -O2 staged.cu -o staged

The best/average timings:
../best_time.sh  ./staged

*/

#include <sys/time.h>
#include <ctype.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

// Number of times to run the test (for better timings accuracy):
#define NTESTS 100

// Number of threads in one block (possible range is 32...1024):
#define BLOCK_SIZE 128

// Total number of threads (total number of elements to process in the kernel):
#define NMAX 1000000

/* Subtract the `struct timeval' values X and Y,
   storing the result in RESULT.
   Return 1 if the difference is negative, otherwise 0.  */

// It messes up with y!

int
timeval_subtract (double *result, struct timeval *x, struct timeval *y)
{
  struct timeval result0;

  /* Perform the carry for the later subtraction by updating y. */
  if (x->tv_usec < y->tv_usec) {
    int nsec = (y->tv_usec - x->tv_usec) / 1000000 + 1;
    y->tv_usec -= 1000000 * nsec;
    y->tv_sec += nsec;
  }
  if (x->tv_usec - y->tv_usec > 1000000) {
    int nsec = (y->tv_usec - x->tv_usec) / 1000000;
    y->tv_usec += 1000000 * nsec;
    y->tv_sec -= nsec;
  }

  /* Compute the time remaining to wait.
     tv_usec is certainly positive. */
  result0.tv_sec = x->tv_sec - y->tv_sec;
  result0.tv_usec = x->tv_usec - y->tv_usec;
  *result = ((double)result0.tv_usec)/1e6 + (double)result0.tv_sec;

  /* Return 1 if result is negative. */
  return x->tv_sec < y->tv_sec;
}


//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


// The kernel:
__global__ void MyKernel (double *d_A, double *d_B)
{
  double x, y, z;

  int i = threadIdx.x + blockDim.x * blockIdx.x;

  if (i >= NMAX)
    return;

  // Some meaningless cpu-intensive computation:

  x = pow(d_A[i], 2.71);
  y = pow(d_A[i], 0.35);
  z = 2*x + 5*y;
  d_B[i] = x + y + z + x*y + x/y + y/z;

  return;
}

//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

int main (int argc,char **argv)
{
  struct timeval  tdr0, tdr1, tdr, tdr01;
  double restime, restime0, restime1;
  int devid, devcount, error, Max_gridsize;
  double *h_A, *h_B, *d_A, *d_B;

  /* find number of device in current "context" */
  cudaGetDevice(&devid);
  /* find how many devices are available */
  if (cudaGetDeviceCount(&devcount) || devcount==0)
    {
      printf ("No CUDA devices!\n");
      exit (1);
    }
  else
    {
      cudaDeviceProp deviceProp; 
      cudaGetDeviceProperties (&deviceProp, devid);
      printf ("Device count, devid: %d %d\n", devcount, devid);
      printf ("Device: %s\n", deviceProp.name);
      printf("[deviceProp.major.deviceProp.minor] = [%d.%d]\n\n", deviceProp.major, deviceProp.minor);
      Max_gridsize = deviceProp.maxGridSize[0];
    }

// Loop to run the timing test multiple times:
int kk;
for (kk=0; kk<NTESTS; kk++)
{

  // Using cudaMallocHost (intead of malloc) to accelerate data copying:  
  // Initial data array on host:
  if (error = cudaMallocHost (&h_A, NMAX*sizeof(double)))
    {
      printf ("Error %d\n", error);
      exit (error);
    }
  // Results array on host:
  if (error = cudaMallocHost (&h_B, NMAX*sizeof(double)))
    {
      printf ("Error %d\n", error);
      exit (error);
    }

  // ALlocating arrays on GPU:
  if (error = cudaMalloc (&d_A, NMAX*sizeof(double)))
    {
      printf ("Error %d\n", error);
      exit (error);
    }
  if (error = cudaMalloc (&d_B, NMAX*sizeof(double)))
    {
      printf ("Error %d\n", error);
      exit (error);
    }


  // Initializing the input array:
  for (int i=0; i<NMAX; i++)
    h_A[i] = (double)rand()/(double)RAND_MAX;
 

  // Number of blocks of threads:
  int Nblocks = (NMAX+BLOCK_SIZE-1) / BLOCK_SIZE;
  if (Nblocks > Max_gridsize)
    {
      printf ("Nblocks > Max_gridsize!  %d  %d\n", Nblocks, Max_gridsize);
      exit (1);
    }

  if (error = cudaDeviceSynchronize())
    {
      printf ("Error %d\n", error);
      exit (error);
    }
  gettimeofday (&tdr0, NULL);
  //--------------------------------------------------------------------------------


  // Copying the data to device (we time it):
  if (error = cudaMemcpy (d_A, h_A, NMAX*sizeof(double), cudaMemcpyHostToDevice))
    {
      printf ("Error %d\n", error);
      exit (error);
    }


  // Intermediate timing, to measure timings separately for copying and kernel execution
  // (Should be removed in the solution code)
  if (error = cudaDeviceSynchronize())
    {
      printf ("Error %d\n", error);
      exit (error);
    }
  gettimeofday (&tdr01, NULL);


  // The kernel call:
  MyKernel <<<Nblocks, BLOCK_SIZE>>> (d_A, d_B);


  //--------------------------------------------------------------------------------
  if (error = cudaDeviceSynchronize())
    {
      printf ("Error %d\n", error);
      exit (error);
    }
  gettimeofday (&tdr1, NULL);
  tdr = tdr0;
  timeval_subtract (&restime, &tdr1, &tdr);

  // Copying the result back to host (we don't time it):
  if (error = cudaMemcpy (h_B, d_B, NMAX*sizeof(double), cudaMemcpyDeviceToHost))
    {
      printf ("Error %d\n", error);
      exit (error);
    }

  if (error = cudaDeviceSynchronize())
    {
      printf ("Error %d\n", error);
      exit (error);
    }

  // Adding up the results, for accuracy/correctness testing:
  double result = 0.0;
  for (int i=0; i<NMAX; i++)
    {
      result += h_B[i];
    }

  tdr = tdr0;
  timeval_subtract (&restime0, &tdr01, &tdr);
  tdr = tdr01;
  timeval_subtract (&restime1, &tdr1, &tdr);
  printf ("Individual timings: %e %e\n", restime0, restime1);

  printf ("Result: %e\n\n", result);
  printf ("Time: %e\n", restime);

  cudaFreeHost (h_A);
  cudaFreeHost (h_B);
  cudaFree (d_A);
  cudaFree (d_B);

} // kk loop

  return 0;

}
