/* CUDA exercise to convert a simple serial code for a brute force
   largest prime number search into CUDA (32-bit, int version). This
   initial code is serial, but it is written as CUDA code for your
   convenience, so should be compiled with nvcc (see below). Your task
   is to convert the serial computation to a kernel computation. In
   the simplest case, use atomicMax to find the globally largest prime
   number.

   All prime numbers can be expressed as 6*k-1 or 6*k+1, k being an
   integer. We provide the range of k to probe as macro parameters
   KMIN and KMAX (see below).

   You should get a speedup ~20x with atomicMax.

Hints:

* It's very convenient to use a two-dimensional grid of blocks,
  defined as "dim3 Nblocks (NBLOCKS, 2, 1);". The second grid
  dimension is used to derive the two values of j=(-1; 1) inside the
  kernel: "int j = 2*blockIdx.y - 1;". This way, there will be only
  one loop inside the kernel - for y.

* When you get a failure (not a prime) inside the y loop, you can exit
  the thread with "return" (no need to use "break").




To compile:

nvcc -arch=sm_20 -O2 primes.cu -o primes

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
#define BLOCK_SIZE 256

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


// Kernel(s) should go here:




int main (int argc,char **argv)
{
  struct timeval  tdr0, tdr1, tdr;
  double restime;
  int devid, devcount, error, success;
  int xmax, ymax, x, y;

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
  for (int k=KMIN; k<=KMAX; k++)
    {
      // testing "-1" and "+1" cases:
      for (int j=-1; j<2; j=j+2)
	{
	  // Prime candidate:
	  x = 6*k + j;
	  // We should be dividing by numbers up to sqrt(x):
	  ymax = (int)ceil(sqrt((double)x));

	  // Primality test:
	  for (y=3; y<=ymax; y=y+2)
	    {
	      // Tpo be a success, the modulus should not be equal to zero:
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
  printf ("%d\n", xmax);
  printf ("Time: %e\n", restime);
  //--------------------------------------------------------------------------------



  return 0;

}
