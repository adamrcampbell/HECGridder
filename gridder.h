
/* 
 * File:   gridder.h
 * Author: adam
 *
 * Created on 1 August 2017, 11:23 AM
 */

#ifndef GRIDDER_H
#define GRIDDER_H

/*--------------------------------------------------------------------
*   STRUCTS
*-------------------------------------------------------------------*/
typedef struct FloatComplex {
    float real;
    float imaginary;
} FloatComplex;

typedef struct Visibility {
    FloatComplex comp;
    float u;
    float v;
    float w;
} Visibility;

typedef struct SpheroidalFunction {
    gsl_vector *itsCoeffs;
    bool itsREven;
    double itsAlpha;
    double itsSum0;
} SpheroidalFunction;

/*--------------------------------------------------------------------
*   ENUMS
*-------------------------------------------------------------------*/
enum kernel {
    RANDOM,
    KAISER, 
    PROLATE
};

/*--------------------------------------------------------------------
*   FUNCTION DEFINITIONS
*-------------------------------------------------------------------*/
void initConfig(void);
void initGridder(void);
void runGridder(void);
void createKernel(int depth);
void checkShaderStatus(GLuint shader);
void checkProgramStatus(GLuint program);
GLuint createShader(GLenum shaderType, const char* shaderSource);
GLuint createProgram(GLuint fragmentShader, GLuint vertexShader);
void timerEvent(int value);
float timedifference_msec(struct timeval t0, struct timeval t1);
void printTimesAverage(struct timeval realStart, int processStart, char description[]);
float getZeroOrderModifiedBessel(float x);
float calculateKaiserPoint(float i);
double calculateKernelWeight(float x);
void initSpheroidal(void);
double calculateSpheroidalPoint(const double nu);
double sumLegendreSeries(const double x, const int m);
void fillHelperMatrix(gsl_matrix *B, const int m);
double fillLegendreCoeffs(const gsl_matrix *B);
void saveGridToFile(void);
void loadVisibilitySamples(void);
void compareToIdealGrid(void);
void calculateSpheroidalCurve(float * nu, int kernelWidth);
void wKernelList(void);
void createWTermLike(int width, FloatComplex wScreen[][width], float w);
void wBeam(int width, FloatComplex wScreen[][width], float fieldOfView, float w, float centerX, float centerY);
int digitize(float w, float wmaxabs);
void calcSpheroidalCurve(float * curve);
void fft2DVectorRadixTransform(int numChannels, const FloatComplex input[][numChannels], FloatComplex output[][numChannels]);
int* calcBitReversedIndices(int n);
void populate3DKernel(void);
void printGrid(void);

FloatComplex complexAdd(FloatComplex x, FloatComplex y);
FloatComplex complexSubtract(FloatComplex x, FloatComplex y);
FloatComplex complexDivide(FloatComplex x, FloatComplex y);
FloatComplex complexMultiply(FloatComplex x, FloatComplex y);
FloatComplex complexExponential(float ph);

#endif /* GRIDDER_H */

