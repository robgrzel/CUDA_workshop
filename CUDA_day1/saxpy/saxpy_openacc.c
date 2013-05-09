/*
SAXPY implementation accelerated with OpenACC.

The compiler offering OpenACC support is PGI.

This code has no CUDA components, so the file extension is .c and not .cu

Compile on monk with:

module unload intel
module load pgi
pgcc -acc -Minfo=accel -fast saxpy_openacc.c

to get timing information, define the following two variables:

export ACC_NOTIFY=1
export PGI_ACC_TIME=1

*/


#include <stdio.h> 
#include <openacc.h>
#include <stdlib.h>

void saxpy_openacc(float *restrict vecY, float *vecX, float alpha, int n)
{
     int i;
#pragma acc kernels
     for (i = 0; i < n; i++)
           vecY[i] = alpha * vecX[i] + vecY[i];
}

void saxpy_cpu(float *vecY, float *vecX, float alpha, int n)
{
     int i;

     for (i = 0; i < n; i++)
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

   memsize = n * sizeof(float);

   /* allocate arrays on host */

   x_host = (float *) malloc(memsize);
   y_host = (float *)malloc(memsize);
   y_shadow = (float *) malloc(memsize);

   /* initialize arrays on host */

   for ( i = 0; i < n; i++)
   {
      x_host[i] = rand() / (float)RAND_MAX;
      y_host[i] = rand() / (float)RAND_MAX;
      y_shadow[i]=y_host[i];
   }

   /* execute openacc accelerated function on GPU */
   saxpy_openacc(y_shadow, x_host, alpha, n);

   /* execute host version (i.e. baseline reference results) */
   saxpy_cpu(y_host, x_host, alpha, n);

   /* check results */
   nerror=0; 
   for(i=0; i < n; i++)
   {
      if(y_shadow[i]!=y_host[i]) nerror=nerror+1;
   }
   printf("test comparison shows %d errors\n",nerror);

   /* free memory */
   free(x_host);
   free(y_host);
   free(y_shadow);

   return 0;   
}


