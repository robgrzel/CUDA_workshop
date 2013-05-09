/*
SAXPY accelerated with CUDA.

In this version CUDA error checks and timing calls have been added.

Compile with:

nvcc -arch=sm_20 -O2 saxpy_cuda_timed.cu  
*/


#include <cuda.h> /* CUDA runtime API */
#include <cstdio> 
#include <sys/time.h>
#include <time.h>


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

void saxpy_cpu(float *vecY, float *vecX, float alpha, int n)
{
     int i;

     for (i = 0; i < n; i++)
           vecY[i] = alpha * vecX[i] + vecY[i];
}

__global__ void saxpy_gpu(float *vecY, float *vecX, float alpha ,int n)
{
   int i;

   i = blockIdx.x * blockDim.x + threadIdx.x;
   if (i<n)
      vecY[i] = alpha * vecX[i] + vecY[i];
}


int main(int argc, char *argv[])
{
   float *x_host, *y_host;   /* arrays for computation on host*/
   float *x_dev, *y_dev;     /* arrays for computation on device */
   float *y_shadow;          /* host-side copy of device results */

   int n = 32*1024;
   float alpha = 0.5f;
   int nerror;
   
   size_t memsize;
   int i, blockSize, nBlocks;

   int error;
   double restime;
   struct timeval  tdr0, tdr1;
   cudaEvent_t start, stop;
   float kernel_timer;

   memsize = n * sizeof(float);

   /* allocate arrays on host */

   x_host = (float *)malloc(memsize);
   y_host = (float *)malloc(memsize);
   y_shadow = (float *)malloc(memsize);

   /* allocate arrays on device */

   if(error = cudaMalloc((void **) &x_dev, memsize))
    {
      printf ("Error in cudaMalloc %d\n", error);
      exit (error);
    }

   if(error = cudaMalloc((void **) &y_dev, memsize))
    {
      printf ("Error in cudaMalloc %d\n", error);
      exit (error);
    }


   /* catch any errors */

   /* initialize arrays on host */

   for ( i = 0; i < n; i++)
   {
      x_host[i] = rand() / (float)RAND_MAX;
      y_host[i] = rand() / (float)RAND_MAX;
   }

   /* copy arrays to device memory (synchronous) */

   gettimeofday (&tdr0, NULL);
  if (error = cudaMemcpy(x_dev, x_host, memsize, cudaMemcpyHostToDevice))
    {
      printf ("Error %d\n", error);
      exit (error);
    }

  if (error = cudaMemcpy(y_dev, y_host, memsize, cudaMemcpyHostToDevice))
    {
      printf ("Error %d\n", error);
      exit (error);
    }

   /* set up device execution configuration */
   blockSize = 512;
   nBlocks = n / blockSize + (n % blockSize > 0);

   /* execute kernel (asynchronous!) */

   cudaEventCreate(&start);
   cudaEventCreate(&stop);
   cudaEventRecord(start, 0);

   saxpy_gpu<<<nBlocks, blockSize>>>(y_dev, x_dev, alpha, n);

   cudaEventRecord(stop, 0);
   cudaEventSynchronize( stop );
   cudaEventElapsedTime( &kernel_timer, start, stop );

   printf("Test Kernel took %f ms\n",kernel_timer);

   /* check success of kernel */
   if(error = cudaGetLastError())
    {
      printf ("Error detected after kernel %d\n", error);
      exit (error);
    }

   cudaEventDestroy(start);
   cudaEventDestroy(stop);

   /* retrieve results from device (synchronous) */
  if (error = cudaMemcpy(y_shadow, y_dev, memsize, cudaMemcpyDeviceToHost))
    {
      printf ("Error %d\n", error);
      exit (error);
    }

   /* check if GPU calculation completed without error */
   if (error = cudaDeviceSynchronize())
    {
      printf ("Error %d\n", error);
      exit (error);
    }

   gettimeofday (&tdr1, NULL);
   timeval_subtract (&restime, &tdr1, &tdr0);
   printf ("gpu kernel and memcopy %e s\n", restime);



   gettimeofday (&tdr0, NULL);

   /* execute host version (i.e. baseline reference results) */
   saxpy_cpu(y_host, x_host, alpha, n);

   gettimeofday (&tdr1, NULL);
   timeval_subtract (&restime, &tdr1, &tdr0);
   printf ("cpu kernel %e s\n", restime);

   /* check results */
   nerror=0; 
   for(i=0; i < n; i++)
   {
      if(y_shadow[i]!=y_host[i]) nerror=nerror+1;
   }
   printf("test comparison shows %d errors\n",nerror);

   /* free memory */
   cudaFree(x_dev);
   cudaFree(y_dev);
   free(x_host);
   free(y_host);
   free(y_shadow);

   return 0;
}


