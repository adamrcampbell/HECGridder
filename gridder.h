
/*
 * 
 * Authors: Dr. Seth Hall, Dr. Andrew Ensor, Adam Campbell
 * Auckland University of Technology - AUT
 * High Performance Laboratory
 * 
 */

#ifndef GRIDDER_H
#define GRIDDER_H

/*--------------------------------------------------------------------
*   ENUMERATION
*-------------------------------------------------------------------*/
enum FragShaderType{FullCube = 0, Radial = 1, Reflect = 2};
/*--------------------------------------------------------------------
*   STRUCTS
*-------------------------------------------------------------------*/
typedef struct Config {
    // General
    unsigned int gridDimension;
    unsigned int renderDimension;
    unsigned int kernelTexSize;
    unsigned int kernelResolutionSize;
    float kernelMaxFullSupport;
    float kernelMinFullSupport;
    unsigned int visibilityCount;
    unsigned int numVisibilityParams;
    bool visibilitiesFromFile;
    double frequencyStartHz;
    bool offsetVisibilities;
    bool useHeavyInterpolation;
    bool accumulateMode;
    bool saveGridToFile;

    // GUI
    unsigned int refreshDelay;
    unsigned int displayDumpTime;
    
    // W Projection
    double cellSizeRad;
    double uvScale;
    double wScale;
    double fieldOfView;
    double wProjectionMaxW;
    unsigned int wProjectNumPlanes;
    double wToMaxSupportRatio;
    
    // Dataset File Locations
    char* visibilitySourceFile;
    char* outputGridReal;
    char* outputGridImag;
    
    enum FragShaderType fragShaderType;
    
} Config;

typedef struct FloatComplex {
    float real;
    float imaginary;
} FloatComplex;

typedef struct DoubleComplex {
    double real;
    double imaginary;
} DoubleComplex;


/*--------------------------------------------------------------------
*   FUNCTION DEFINITIONS
*-------------------------------------------------------------------*/
void initConfig(char** argv);

void initGridder(void);

void runGridder(void);

void checkShaderStatus(GLuint shader);

void checkProgramStatus(GLuint program);

void setShaderUniforms(void);

GLuint createShader(GLenum shaderType, const char* shaderSource);

GLuint createProgram(GLuint fragmentShader, GLuint vertexShader);

void timerEvent(int value);

void saveGridToFile(int support);

void createWProjectionPlanes(FloatComplex *wTextures);

void createPhaseScreen(int resolutionFullSupport, DoubleComplex *screen, 
        double w, double fieldOfView, int wFullSupport);

double calcSpheroidalWeight(double nu);

void inverseFFT2dVectorRadixTransform(int numChannels, DoubleComplex *input, 
        DoubleComplex *output);

void calcBitReversedIndices(int n, int* indices);

void fft2dShift(int n, DoubleComplex *input, DoubleComplex *shifted);

float calcInterpolateShift(float index, float width);

void saveKernelToFile(char* filename, float w, int support, 
        DoubleComplex* data);

void saveRadialKernelsToFile(char* filename, int support, int wPlanes, 
        FloatComplex* data);

int calcRelativeIndex(double x, double width);

double calcSphrShift(double index, double width);

void interpolateKernel(DoubleComplex *source, DoubleComplex *destination, 
    int sourceSupport, int destinationSupport);

void getBicubicNeighbours(double rowShift, double colShift, DoubleComplex *n, 
        double *rs, double *cs,
        int sourceSupport, DoubleComplex *source);

DoubleComplex interpolateSample(DoubleComplex z0, DoubleComplex z1, 
    DoubleComplex z2, DoubleComplex z3, double x0, double x1, double x2,
    double x3, double h, double x);

DoubleComplex complexAdd(DoubleComplex x, DoubleComplex y);

DoubleComplex complexSubtract(DoubleComplex x, DoubleComplex y);

DoubleComplex complexMultiply(DoubleComplex x, DoubleComplex y);

DoubleComplex complexConjugateExp(double ph);

DoubleComplex complexScale(DoubleComplex z, double scalar);

void normalizeKernel(DoubleComplex *kernel, int textureSupport, 
        int wFullSupport);

void normalizeKernelRadial(DoubleComplex *kernel, int resolution, int support);

float calcAndrewShift(int index, int fullSupport);

#endif /* GRIDDER_H */
