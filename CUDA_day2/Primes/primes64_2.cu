/* 

Long long int (64 bit) solution.

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
#define KMAX 100100000

// Number of threads in one block (possible range is 32...1024):
#define BLOCK_SIZE 256

// Target number of blocks used in a single kernel launch (should be >=14)
#define NBLOCKS 560

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


// Initialization kernel:
__global__ void InitKernel ()
{
  d_xmax = 0;
  return;
}



/* The main kernel:
 x0 is the minimum prime candidate value for the current kernel launch
 (6*k0-1, where k0 is the initial k value for the current kernel)
 This is needed to be able to use atomicMax (only defined for int) to
 find the maximum of long long int primes.
*/ 
__global__ void MyKernel (long long int k0, long long int x0)
{
  long long int x, y, ymax, k, j;
  int dx;


  // Global index is shifted by k0:
  k = k0 + threadIdx.x + blockDim.x * blockIdx.x;
  if (k > KMAX)
    return;

  j = 2*blockIdx.y - 1;

  // Prime candidate:
  x = 6*k + j;
  // We should be dividing by numbers up to sqrt(x):
  ymax = (long long int)ceil(sqrt((double)x));

  // Primality test:
  for (y=3; y<=ymax; y=y+2)
    {
      // To be a success, the modulus should not be equal to zero:
      if (x%y == 0)
	return;
    }

  // We get here only if x is a prime number

  // This is needed as atomicMax doesn't work with long long int:
  dx = (int)(x - x0);

  // Storing the globally largest dx:
  atomicMax (&d_xmax, dx);

  return;
}




int main (int argc,char **argv)
{
  struct timeval  tdr0, tdr1, tdr;
  double restime;
  int devid, devcount, error, h_xmax;
  long long int xmax, Delta_k, k1, k0, x0, Xmax;

  printf ("long long int size on the host: %d bit\n", 8*sizeof(long long int));

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

  k0 = KMIN;
  Xmax = 0;

  // Range of k explored in a single kernel launch:
  Delta_k = NBLOCKS * BLOCK_SIZE;
  dim3 Nblocks (NBLOCKS, 2, 1);

  k1 = KMAX - Delta_k;

  while (k0 < KMAX)
    {

      // Smallest prime candidate for the current kernel:
      x0 = 6*k0 - 1;

      if (k0 > k1)
	// We are dealing with the last (truncated) kernel launch
	dim3 Nblocks ((KMAX-k0+BLOCK_SIZE-1)/BLOCK_SIZE, 2, 1);
      else
	// Regular (complete) kernel launch
	// It is very convenient to create blocks on a 2D grid, with the second dimension
	// of size two corresponding to "-1" and "+1" cases:
	dim3 Nblocks (NBLOCKS, 2, 1);

      // Initializing d_xmax:
      InitKernel <<<1, 1>>> ();

      // The kernel call:
      MyKernel <<<Nblocks, BLOCK_SIZE>>> (k0, x0);

      if (error = cudaDeviceSynchronize())
	{
	  printf ("Error %d\n", error);
	  exit (error);
	}

      // Copying the current result to host:
      if (error = cudaMemcpyFromSymbol (&h_xmax, d_xmax, sizeof(int), 0, cudaMemcpyDeviceToHost))
	{
	  printf ("Error %d\n", error);
	  exit (error);
	}
      
      // Largest prime found in the current kernel (0 if none found):
      xmax = x0 + h_xmax;

      // Globally largest prime:
      if (xmax > Xmax)
	Xmax = xmax;

      // When code compiled with -DTIME (for timing purposes), it will not print out the intermediate results:
#ifndef TIME
      long long int kmax;
      if (k0+Delta_k > KMAX)
	kmax = KMAX;
      else
	kmax = k0+Delta_k-1;
      printf ("Range of k: %ld ... %ld; largest prime: %ld\n", k0, kmax, xmax);
#endif

      k0 = k0 + Delta_k;

    }

  if (error = cudaDeviceSynchronize())
    {
      printf ("Error %d\n", error);
      exit (error);
    }
  gettimeofday (&tdr1, NULL);
  tdr = tdr0;
  timeval_subtract (&restime, &tdr1, &tdr);
  printf ("\n%ld\n", Xmax);
  printf ("Time: %e\n", restime);
  //--------------------------------------------------------------------------------



  return 0;

}
