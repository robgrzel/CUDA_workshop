/*
Code adapted from book "CUDA by Example: An Introduction to General-Purpose GPU Programming" 

This code computes a visualization of the Julia set.  Two-dimenansional "bitman" data which can be plotted is computed by the function kernel.

The data can be viewed with gnuplot.

The Julia set iteration is:

z= z**2 + C

If it converges, then the initial point z is in the Julia set.

 
*/


#include <stdio.h>

#define DIM 1000

int julia( int x, int y ) { 
    const float scale = 1.5;
    float jx = scale * (float)(DIM/2 - x)/(DIM/2);
    float jy = scale * (float)(DIM/2 - y)/(DIM/2);

    float cr=-0.8f;
    float ci=0.156f;

    float ar=jx;
    float ai=jy;
    float artmp;

    int i = 0;
    for (i=0; i<200; i++) {

        artmp = ar;
        ar =(ar*ar-ai*ai) +cr;
        ai = 2.0f*artmp*ai + ci;

        if ( (ar*ar+ai*ai) > 1000)
            return 0;
    }

    return 1;
}

void kernel( int *arr ){
    for (int y=0; y<DIM; y++) {
        for (int x=0; x<DIM; x++) {
            int offset = x + y * DIM;

            int juliaValue = julia( x, y );
            arr[offset] = juliaValue;
        }
    }
 }

int main( void ) {
    int arr[DIM*DIM];
    FILE *out;
    kernel(arr);

    out = fopen( "julia.dat", "w" );
    for (int y=0; y<DIM; y++) {
        for (int x=0; x<DIM; x++) {
            int offset = x + y * DIM;
                    if(arr[offset]==1){
                       fprintf(out,"%d %d \n",x,y);  }
       } 
    } 
    fclose(out);


}

