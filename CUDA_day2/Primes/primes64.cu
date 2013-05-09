/* CUDA exercise to convert a simple serial code for a brute force
   largest prime number search into CUDA. This initial code is serial,
   but it is written as CUDA code for your convenience, so should be
   compiled with nvcc (see below). Your task is to convert the serial
   computation to a kernel computation. In the simplest case, use
   atomicMax to find the globally largest prime number.

   All prime numbers can be expressed as 6*k-1 or 6*k+1, k being an
   integer. We provide the range of k to probe as macro parameters
   KMIN and KMAX (see below).

   You should get a speedup ~22 (with KMIN=100000000, KMAX=100100000,
   BLOCK_SIZE=256, and default number of blocks per kernel NBLOCKS=560).

   This is a 64-bit (long long int, instead of int) version - so in principle
   you can find primes up to 2^64-1, or 1.8e19.


Hints:

* You can still use atomicMax, even in this 64-bit version, if you use
  it not with prime numbers themselves (xmax), but with differences
  between the prime number and the starting prime number candidate
  value for the current kernel (__device__ int d_xmax), which should
  fit in a 32-bit integer for any realistic size kernel.

* On the host, computation should be organized in a while loop, which
  sets the initial prime candidate value for the loop, x0, computes number
  of blocks for the main kernel, initializes d_xmax in a single-thread
  kernel, and then submit the main kernel to device. Then you should copy
  the current value of d_xmax back to the host, and compute the largest
  found prime (this time using 64-bit integers) for the loop, as x0+d_xmax.

* It's very convenient to use a two-dimensional grid of blocks,
  defined as "dim3 Nblocks (NBLOCKS, 2, 1);". The second grid
  dimension is used to derive the two values of j=(-1; 1) inside the
  kernel: "int j = 2*blockIdx.y - 1;". This way, there will be only
  one loop inside the kernel - for y.

* When you get a failure (not a prime) inside the y loop, you can exit
  the thread with "return" (no need to use "break").




To compile:

nvcc -arch=sm_20 -O2 primes64.cu -o primes64

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


// Kernel(s) should go here:




int main (int argc,char **argv)
{
  struct timeval  tdr0, tdr1, tdr;
  double restime;
  int devid, devcount, error, success;
  long long int k, j, x, xmax, y, ymax;

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


  // This serial computation will have to be replaced by calls to kernel(s):
  xmax = 0;
  for (k=KMIN; k<=KMAX; k++)
    {
      // testing "-1" and "+1" cases:
      for (j=-1; j<2; j=j+2)
	{
	  // Prime candidate:
	  x = 6*k + j;
	  // We should be dividing by numbers up to sqrt(x):
	  ymax = (long long int)ceil(sqrt((double)x));

	  // Primality test:
	  for (y=3; y<=ymax; y=y+2)
	    {
	      // To be a success, the modulus should not be equal to zero:
	      success = x % y;
	      if (!success)
		break;
	    }

	  if (success && x > xmax)
	    {
	      xmax = x;
	    }
	}
    }

  if (error = cudaDeviceSynchronize())
    {
      printf ("Error %d\n", error);
      exit (error);
    }
  gettimeofday (&tdr1, NULL);
  tdr = tdr0;
  timeval_subtract (&restime, &tdr1, &tdr);
  printf ("%ld\n", xmax);
  printf ("Time: %e\n", restime);
  //--------------------------------------------------------------------------------



  return 0;

}
