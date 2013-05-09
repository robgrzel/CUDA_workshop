/* Small CUDA exercise to detect bad (non-coalesced) memory access,
   and to make it coalesced.

For NMAX=1000000, STRIDE=30, BLOCK_SIZE=128, the speedup (from
non-coalesced to coalesced versions of the code) should be ~3.8x.

Make sure that the "Result:" value printed by the code is (almost)
identical in both original and modified versions of the code. If not,
you have a bug!


To compile:

nvcc -O2 -arch=sm_20  coalesce.cu -o coalesce

The best/average timings:
../best_time.sh  ./coalesce

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
#define NTESTS 10

// Number of threads in one block (possible range is 32...1024):
#define BLOCK_SIZE 128

// Size of the array:
#define NMAX 1000000

// "Stride" (the smaller dimension) of the 2D array:
#define STRIDE 30

// Input 2D array:
float h_A[NMAX][STRIDE];
__device__ float d_A[NMAX][STRIDE];

// The result will go here:
__device__ float d_B[NMAX];
float h_B[NMAX];


/* Subtract the `struct timeval' values X and Y,
   storing the result in RESULT.
   Return 1 if the difference is negative, otherwise 0.  */

// It messes up with y!

int
timeval_subtract (float *result, struct timeval *x, struct timeval *y)
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
  *result = ((float)result0.tv_usec)/1e6 + (float)result0.tv_sec;

  /* Return 1 if result is negative. */
  return x->tv_sec < y->tv_sec;
}


//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


// The kernel:
__global__ void MyKernel ()
{
  // Global index within a block:
  int i = threadIdx.x + blockDim.x * blockIdx.x;

  if (i >= NMAX)
    return;

  // Second array index is a function of the blockID:
  int j = blockIdx.x % STRIDE;

  d_B[i] = pow(d_A[i][j], 0.73f);

  return;
}

//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


int main (int argc,char **argv)
{
  struct timeval  tdr0, tdr1, tdr;
  float restime;
  int devid, devcount, error, Max_gridsize;

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

  // Initializing the structure elements:
  for (int i=0; i<NMAX; i++)
    for (int j=0; j<STRIDE; j++)
      h_A[i][j] = (float)rand()/(float)RAND_MAX;

  // Copying the data to device:
  if (error = cudaMemcpyToSymbol( d_A, h_A, sizeof(h_A), 0, cudaMemcpyHostToDevice))
    {
      printf ("Error %d\n", error);
      exit (error);
    }

  // Number of blocks of threads:
  int Nblocks = (NMAX+BLOCK_SIZE-1) / BLOCK_SIZE;
  if (Nblocks > Max_gridsize)
    {
      printf ("Nblocks > Max_gridsize!  %d  %d\n", Nblocks, Max_gridsize);
      exit (1);
    }

  //  Only the code between the two horizontal lines is timed:
  if (error = cudaDeviceSynchronize())
    {
      printf ("Error %d\n", error);
      exit (error);
    }
  gettimeofday (&tdr0, NULL);
  //--------------------------------------------------------------------------------


  // The kernel call:
  MyKernel <<<Nblocks, BLOCK_SIZE>>> ();


  //--------------------------------------------------------------------------------
  if (error = cudaDeviceSynchronize())
    {
      printf ("Error %d\n", error);
      exit (error);
    }
  gettimeofday (&tdr1, NULL);

  tdr = tdr0;
  timeval_subtract (&restime, &tdr1, &tdr);

  // Copying the result back to host:
  if (error = cudaMemcpyFromSymbol (h_B, d_B, sizeof(h_B), 0, cudaMemcpyDeviceToHost))
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

  printf ("Result: %e\n\n", result);
  printf ("Time: %e\n", restime);

} // kk loop

  return 0;

}
