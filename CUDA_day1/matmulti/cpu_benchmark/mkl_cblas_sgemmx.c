/*******************************************************************************
!Adapted from Intel SDK example
!cc mkl_cblas_sgemmx.c -Wl,--start-group -lmkl_intel_lp64 -lmkl_sequential -lmkl_core -Wl,--end-group -lpthread -lm -o sgemm.x
! with threaded libraries
!cc mkl_cblas_sgemmx.c -Wl,--start-group -lmkl_intel_lp64 -lmkl_intel_thread -lmkl_core -Wl,--end-group -lpthread -lm -liomp5 -o sgemm_threaded.x
! a is m by k
! b is k by n
! c is m by n 
! matrices stored in column major order
!******************************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <time.h>

#include "mkl_example.h"

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
 *      tv_usec is certainly positive. */
  result0.tv_sec = x->tv_sec - y->tv_sec;
  result0.tv_usec = x->tv_usec - y->tv_usec;
  *result = ((double)result0.tv_usec)/1e6 + (double)result0.tv_sec;

  /* Return 1 if result is negative. */
  return x->tv_sec < y->tv_sec;
}




int main(int argc, char *argv[])
{
      FILE *in_file;
      char *in_file_name;

      MKL_INT         m, n, k;
      MKL_INT         lda, ldb, ldc;
      MKL_INT         rmaxa, cmaxa, rmaxb, cmaxb, rmaxc, cmaxc;
      float           alpha, beta;
      float          *a, *b, *c;
      CBLAS_ORDER     order;
      CBLAS_TRANSPOSE transA, transB;
      MKL_INT         ma, na, mb, nb;
      MKL_INT         i,j;
      struct timeval  tdr0, tdr1;
      double restime;

      printf("\n     C B L A S _ S G E M M  EXAMPLE PROGRAM\n");

/*       Get input parameters and data                         */ 

      m = 32*512;
      n = 32*512;
      k = 32;


      lda=m;
      ldb=k;
      ldc=m;

      alpha = 1.0f;  
      beta = 0.0f;

      transA = CblasNoTrans;
      transB = CblasNoTrans;
      order = CblasColMajor;


      a = (float *)calloc( m*k, sizeof( float ) );
      b = (float *)calloc( k*n, sizeof( float ) );
      c = (float *)calloc( m*n, sizeof( float ) );

      if( a == NULL || b == NULL || c == NULL ) {
          printf( "\n Can't allocate memory for arrays\n");
          return 1;
      }

      for(i = 0;i<m*k; i++){
           a[i]=rand() / (float)RAND_MAX;
      }

      for(i = 0;i<k*n;i++){
           b[i] = rand() / (float)RAND_MAX; 
      }

      for(i = 0;i<m*n;i++){
           c[i]= rand() / (float)RAND_MAX;
       }

/*      Call SGEMM subroutine ( C Interface )                  */

  gettimeofday (&tdr0, NULL);

      cblas_sgemm(order, transA, transB, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);

  gettimeofday (&tdr1, NULL);
  timeval_subtract (&restime, &tdr1, &tdr0);
  printf("multiplying single-precision matrices (%d x %d) and (%d x %d) via cblas_sgemm took \n",m,k,k,n); 
  printf ("%e seconds\n", restime);


      free(a);
      free(b);
      free(c);

      return 0;
}

