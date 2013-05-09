/* using column major storage 

Compile on monk with:

nvcc -arch=sm_20 -O2 matmulti_CUDA.cu -o matmulti_CUDA.x  

Program must be run on system with a working CUDA GPU 






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

__global__ void simpleMultiply_gpu(float *a, float *b, float *c ,int N)
{

  int col=blockIdx.y*blockDim.y + threadIdx.y;
  int row=blockIdx.x*blockDim.x + threadIdx.x;

  float sum = 0.0f;

  for (int i=0; i < TILE_DIM; i++) {
      sum+= a[row+i*N]*b[i+col*TILE_DIM];
  }
  c[row+col*N]=sum;
}

__global__ void sharedmemoryMultiply_gpu(float *a, float *b, float *c ,int N)
{
  __shared__ float aTile[TILE_DIM*TILE_DIM], bTile[TILE_DIM*TILE_DIM];

  int col=blockIdx.y*blockDim.y + threadIdx.y;
  int row=blockIdx.x*blockDim.x + threadIdx.x;

  float sum = 0.0f;

  aTile[threadIdx.y*TILE_DIM+threadIdx.x] = a[row+threadIdx.y*N];
  bTile[threadIdx.y*TILE_DIM+threadIdx.x] = b[threadIdx.x+col*TILE_DIM];
  __syncthreads();
  for (int i=0; i < TILE_DIM; i++) {
      sum+= aTile[threadIdx.x+ i*TILE_DIM]*bTile[i+threadIdx.y*TILE_DIM];
  }
  c[row+col*N]=sum;
}


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
   dim3 gridDef(n/TILE_DIM,n/TILE_DIM);
   dim3 blockDef(TILE_DIM,TILE_DIM,1);

   struct timeval  tdr0, tdr1;
   int error;
   cudaEvent_t start, stop;

   /*  find compute device an initialize it */
   /* add device detection */

   memsize_input = n * TILE_DIM * sizeof(float);
   memsize_output = n * n * sizeof(float);
   /* allocate arrays on host */

   if(error = cudaMallocHost((void **) &a_host, memsize_input))
{
      printf ("Error in cudaMallocHost %d\n", error);
      exit (error);
}

   if(error = cudaMallocHost((void **) &b_host, memsize_input))
{
      printf ("Error in cudaMallocHost %d\n", error);
      exit (error);
}

   if(error = cudaMallocHost((void **) &c_host, memsize_output))
{
      printf ("Error in cudaMallocHost %d\n", error);
      exit (error);
}

   if(error = cudaMallocHost((void **) &c_shadow, memsize_output))
{
      printf ("Error in cudaMallocHost %d\n", error);
      exit (error);
}



   /* allocate arrays on device */

   if(error = cudaMalloc((void **) &a_dev, memsize_input))
    {
      printf ("Error in cudaMalloc %d\n", error);
      exit (error);
    }

   if(error = cudaMalloc((void **) &b_dev, memsize_input))
    {
      printf ("Error in cudaMalloc %d\n", error);
      exit (error);
    }

   if(error = cudaMalloc((void **) &c_dev, memsize_output))
    {
      printf ("Error in cudaMalloc %d\n", error);
      exit (error);
    }

   /* initialize arrays on host */

   for ( int i = 0; i < n*TILE_DIM; i++)
   {
      a_host[i] = rand() / (float)RAND_MAX;
      b_host[i] = rand() / (float)RAND_MAX;
   }

   /* copy arrays to device memory (synchronous) */

  gettimeofday (&tdr0, NULL);

  if (error = cudaMemcpy(a_dev, a_host, memsize_input, cudaMemcpyHostToDevice))
    {
      printf ("Error %d\n", error);
      exit (error);
    }

  if (error = cudaMemcpy(b_dev, b_host, memsize_input, cudaMemcpyHostToDevice))
    {
      printf ("Error %d\n", error);
      exit (error);
    }

  float kernel_timer;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start, 0);

  simpleMultiply_gpu<<<gridDef,blockDef>>>(a_dev, b_dev, c_dev, n);

  cudaEventRecord(stop, 0);
  cudaEventSynchronize( stop ); 
  cudaEventElapsedTime( &kernel_timer, start, stop );

  printf("Test Kernel took %f ms\n",kernel_timer);
  printf ("GFlops %f\n",  (float)(n*n)*(2.0f*(float)TILE_DIM-1.0f)/kernel_timer/1000000.0f);

   /* retrieve results from device (synchronous) */
  if (error =  cudaMemcpy(c_shadow, c_dev, memsize_output, cudaMemcpyDeviceToHost))
    {
      printf ("Error %d\n", error);
      exit (error);
    }

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
   cudaFree(a_host);
   cudaFree(b_host);
   cudaFree(c_host);
   cudaFree(c_shadow);

}


