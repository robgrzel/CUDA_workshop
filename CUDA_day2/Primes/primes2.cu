/* 32-bit (int) largest prime finder.

Atomic reduction solution.

*/

#include <sys/time.h>
#include <ctype.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

// Range of k-numbers for primes search:
#define KMIN 100000000
// Should be smaller than 357,913,941 (because we are using signed int)
#define KMAX 100100000

// Number of threads in one block (possible range is 32...1024):
#define BLOCK_SIZE 128

// Number of blocks to run:
#define NBLOCKS (KMAX-KMIN+BLOCK_SIZE)/BLOCK_SIZE

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


__device__ int d_xmax;

// Kernel(s) should go here:
// The kernel:
__global__ void MyKernel ()
{
  int x, y, ymax;

  // Global index is shifted by KMIN:
  int k = KMIN + threadIdx.x + blockDim.x * blockIdx.x;
  if (k > KMAX)
    return;

  int j = 2*blockIdx.y - 1;

  // Prime candidate:
  x = 6*k + j;
  // We should be dividing by numbers up to sqrt(x):
  ymax = (int)ceil(sqrt((double)x));

  // Primality test:
  for (y=3; y<=ymax; y=y+2)
    {
      // To be a success, the modulus should not be equal to zero:
      if (x%y == 0)
	return;
    }

  // We get here only if x is a prime number

  // Storing the globally largest prime:
  atomicMax (&d_xmax, x);

  return;
}




int main (int argc,char **argv)
{
  struct timeval  tdr0, tdr1, tdr;
  double restime;
  int devid, devcount, error, xmax;

  if (BLOCK_SIZE>1024)
    {
      printf ("Bad BLOCK_SIZE: %d\n", BLOCK_SIZE);
      exit (1);
    }

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
    }

  //--------------------------------------------------------------------------------
  if (error = cudaDeviceSynchronize())
    {
      printf ("Error %d\n", error);
      exit (error);
    }
  gettimeofday (&tdr0, NULL);

  // It is very convenient to create blocks on a 2D grid, with the second dimension
  // of size two corresponding to "-1" and "+1" cases:
  dim3 Nblocks (NBLOCKS, 2, 1);


  // The kernel call:
  MyKernel <<<Nblocks, BLOCK_SIZE>>> ();


  // Copying the result back to host:
  if (error = cudaMemcpyFromSymbol (&xmax, d_xmax, sizeof(int), 0, cudaMemcpyDeviceToHost))
    {
      printf ("Error %d\n", error);
      exit (error);
    }

  if (error = cudaDeviceSynchronize())
    {
      printf ("Error %d\n", error);
      exit (error);
    }
  gettimeofday (&tdr1, NULL);
  tdr = tdr0;
  timeval_subtract (&restime, &tdr1, &tdr);
  printf ("%d\n", xmax);
  printf ("Time: %e\n", restime);
  //--------------------------------------------------------------------------------



  return 0;

}
