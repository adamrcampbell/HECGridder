
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
typedef struct Complex {
    float real;
    float imaginary;
} Complex;

typedef struct Visibility {
    Complex complex;
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
void createKernel(void);
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

#endif /* GRIDDER_H */

