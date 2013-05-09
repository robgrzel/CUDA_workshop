/*
NOTE: this code can be used as a starting point for your work.

Matrix multiplication code AB=C

where

A = n by TILE_DIM
B = TILE_DIM by n
C = matrix

All these are single precision matrices.

NOTE: column-major storage is used, for easy compatibility with CUBLAS routines.


Compile on monk with:

nvcc -arch=sm_20 -O2 matmulti_CUDA.cu -o matmulti_start.x  

Program must be run on system with a working CUDA GPU. 



*/

#include <cuda.h> /* CUDA runtime API */
#include <cstdio> 
#include <math.h>
#include <sys/time.h>
#include <time.h>
#include <cublas_v2.h>

#define TILE_DIM 32 

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

int check_float(float x,float y)
{
float rel_err; 
float tol=0.00001f;

rel_err= fabsf(x-y) / y;

if(rel_err>tol) return 0;
else return 1;

}

void simpleMultiply_cpu(float *a, float *b, float *c ,int N)
{
   float sum;

   for (int col=0; col < N ; col++ ){
     for (int row=0; row < N ; row++ ){

       sum = 0.0f;
       for (int i=0; i < TILE_DIM; i++) {
          sum+= a[row+i*N]*b[i+col*TILE_DIM];
       }
       c[row+col*N]=sum;
     }
   
   }
}

//__global__ void simpleMultiply_gpu(...)  HINT: GPU kernel
//{
//
//}



int main(int argc, char *argv[])
{
   float *a_host, *b_host, *c_host;   /* arrays for computation on host*/
   float *a_dev, *b_dev, *c_dev;     /* arrays for computation on device */
   float *c_shadow;          /* host-side copy of device results */

   int n = 512 * TILE_DIM ;
   int nerror;
   double restime;

   size_t memsize_input,memsize_output;

   /* set up device execution configuration */
//   dim3 gridDef(...,...,...);   HINT: define blocks and grids
//   dim3 blockDef(...,...,...);

   struct timeval  tdr0, tdr1;
   int error;
   cudaEvent_t start, stop;

   /*  find compute device an initialize it */
   /* add device detection */

   memsize_input = n * TILE_DIM * sizeof(float);
   memsize_output = n * n * sizeof(float);
   /* allocate arrays on host */

/* HINT: change these to cudaMallocHost to use pinned host memory for better performance */
   a_host = (float *)malloc(memsize_input);
   b_host = (float *)malloc(memsize_input);
   c_host = (float *)malloc(memsize_output);


   c_shadow = (float *)malloc(memsize_output);

   /* allocate arrays on device */

//   if(error = cudaMalloc())  HINT: allocate memory on host
//    {
//      printf ("Error in cudaMalloc %d\n", error);
//      exit (error);
//    }


   /* initialize arrays on host */

   for ( int i = 0; i < n*TILE_DIM; i++)
   {
      a_host[i] = rand() / (float)RAND_MAX;
      b_host[i] = rand() / (float)RAND_MAX;
   }

   /* copy arrays to device memory (synchronous) */

  gettimeofday (&tdr0, NULL);

//  if (error = cudaMemcpy(...))      HINT: copy memory to device
//    {
//      printf ("Error %d\n", error);
//      exit (error);
//    }


  float kernel_timer;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start, 0);

//   simpleMultiply_gpu<<<...,...>>>(...);    your kernel call goes here

  cudaEventRecord(stop, 0);
  cudaEventSynchronize( stop ); 
  cudaEventElapsedTime( &kernel_timer, start, stop );

  printf("Test Kernel took %f ms\n",kernel_timer);
  printf ("GFlops %f\n",  (float)(n*n)*(2.0f*(float)TILE_DIM-1.0f)/kernel_timer/1000000.0f);

   /* retrieve results from device (synchronous) */
//  if (error =  cudaMemcpy(...))    HINT: copy memory from device
//    {
//      printf ("Error %d\n", error);
//      exit (error);
//    }

  gettimeofday (&tdr1, NULL);
  timeval_subtract (&restime, &tdr1, &tdr0);
  printf ("gpu kernel and memcopy%e\n", restime);
  printf ("GFlops %e\n",  (float)(n*n)*(2.0f*(float)TILE_DIM-1.0f)/restime/1000000000.0f);

  gettimeofday (&tdr0, NULL);
   /* execute host version (i.e. baseline reference results) */
   simpleMultiply_cpu(a_host, b_host, c_host, n);

  gettimeofday (&tdr1, NULL);
  timeval_subtract (&restime, &tdr1, &tdr0);
  printf ("cpu kernel %e\n", restime);
  printf ("GFlops %e\n",  (float)(n*n)*(2.0f*(float)TILE_DIM-1.0f)/restime/1000000000.0f);

   nerror=0; 
   for(int i=0; i < n*n; i++)
   {
      if(check_float(c_shadow[i],c_host[i])==0) nerror=nerror+1;
   }
   printf("test comparison shows %d errors\n",nerror);


   cudaEventDestroy(start);
   cudaEventDestroy(stop);
   cudaFree(a_dev);
   cudaFree(b_dev);
   cudaFree(c_dev);
   free(a_host);
   free(b_host);
   free(c_host);
   free(c_shadow);

}


