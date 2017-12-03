
#ifndef GRIDDER_H
#define GRIDDER_H

/*--------------------------------------------------------------------
*   STRUCTS
*-------------------------------------------------------------------*/
typedef struct Config {
    // General
    float gridDimension;
    unsigned int kernelTexSize;
    unsigned int kernelMaxFullSupport;
    unsigned int kernelMinFullSupport;
    unsigned int visibilityCount;
    unsigned int numVisibilityParams;
    bool visibilitiesFromFile;
    char* visibilitySourceFile;

    // GUI
    unsigned int refreshDelay;
    unsigned int displayDumpTime;
    
    // Prolate Spheroidal
    double prolateC;
    double prolateAlpha;
    unsigned int prolateNumTerms;
    // Consider what to do with Spheroidal struct
    // as not required with new method of spheroidal
    // calculation (no GSL lib required)
    
    // W Projection
    float cellSize;
    float uvScale;
    float wScale;
    float fieldOfView;
    float wProjectionStep;
    float wProjectionMaxPlane;
    float wProjectionMaxW;
    unsigned int wProjectNumPlanes;
} Config;

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
void initSpheroidal(void);
double calculateSpheroidalPoint(const double nu);
double sumLegendreSeries(const double x, const int m);
void fillHelperMatrix(gsl_matrix *B, const int m);
double fillLegendreCoeffs(const gsl_matrix *B);
void saveGridToFile(void);
void loadVisibilitySamples(void);
void compareToIdealGrid(void);
void calculateSpheroidalCurve(float * nu, int kernelWidth);
void createWPlanes(void);
void createWTermLike(int width, FloatComplex wScreen[][width], float w);
void wBeam(int width, FloatComplex wScreen[][width], float w, float centerX, float centerY, float fieldOfView);
void calcSpheroidalCurve(float * curve);
void fft2dVectorRadixTransform(int numChannels, const FloatComplex input[][numChannels], FloatComplex output[][numChannels]);
void fft2dShift(int numChannels, FloatComplex input[][numChannels], FloatComplex output[][numChannels]);
void fft2dInverseShift(int numChannels, FloatComplex input[][numChannels], FloatComplex output[][numChannels]);
void calcBitReversedIndices(int n, int* indices);
void printGrid(void);

FloatComplex complexAdd(FloatComplex x, FloatComplex y);
FloatComplex complexSubtract(FloatComplex x, FloatComplex y);
FloatComplex complexMultiply(FloatComplex x, FloatComplex y);
FloatComplex complexExponential(float ph);

#endif /* GRIDDER_H */
