/* 

The solution.

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

// Number of chunks (NMAX should be dividable by NCHUNKS):
#define NCHUNKS 10


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
__global__ void MyKernel (double *d_A, double *d_B, int ind, int Ntot)
{
  double x, y, z;

  // Local index:
  int i0 = threadIdx.x + blockDim.x * blockIdx.x;

  if (i0 >= Ntot)
    return;

  // Global index is shifted by ind:
  int i = ind + i0;
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
  struct timeval  tdr0, tdr1, tdr;
  double restime;
  int devid, devcount, error, Max_gridsize;
  double *h_A, *h_B, *d_A, *d_B;
  cudaStream_t  ID[2];

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
    {
      h_A[i] = (double)rand()/(double)RAND_MAX;
    }

  // Creating streams:
  for (int i = 0; i < 2; ++i)
    cudaStreamCreate (&ID[i]);

  // Number of threads in a chunk:
  int Ntot = NMAX / NCHUNKS;

  // Number of blocks of threads in a chunk:
  int Nblocks = (Ntot+BLOCK_SIZE-1) / BLOCK_SIZE;
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


  for (int j=0; j<NCHUNKS; j++)
    {

      // Starting global index for this chunk:
      int ind = Ntot * j;

      // Copying the j-th chunk to device - asynchronous relative to the host and stream ID[0] kernel (processing the previous chunk in parallel)
      if (error = cudaMemcpyAsync (&d_A[ind], &h_A[ind], Ntot*sizeof(double), cudaMemcpyHostToDevice, ID[1]))
	{
	  printf ("Error %d\n", error);
	  exit (error);
	}
      
      // This global synchronization between both streams and host ensures that the kernel can only start 
      // when the previous chunk copying is finished.
      // This also ensures that at j=0 kernel will not start untill the first chunk is copied.
      if (error = cudaDeviceSynchronize())
	{
	  printf ("Error %d\n", error);
	  exit (error);
	} 

      // The kernel call:
      MyKernel <<<Nblocks, BLOCK_SIZE, 0, ID[0]>>> (d_A, d_B, ind, Ntot);
      
    }


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

  printf ("Result: %e\n\n", result);
  printf ("Time: %e\n", restime);

  cudaFreeHost (h_A);
  cudaFreeHost (h_B);
  cudaFree (d_A);
  cudaFree (d_B);

} // kk loop

  return 0;

}
