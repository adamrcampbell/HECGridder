
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
    bool compareToOxfordGrid;
    bool useHeavyInterpolation;
    bool accumulateMode;
    bool saveGridToFile;

    // GUI
    unsigned int refreshDelay;
    unsigned int displayDumpTime;
    float graphicMultiplier;
    
    // W Projection
    double cellSizeRad;
    double uvScale;
    double wScale;
    double fieldOfView;
    double wProjectionMaxW;
    unsigned int wProjectNumPlanes;
    double wToMaxSupportRatio;
    
    // Dataset File Locations
    char* inputGridComparisonReal;
    char* inputGridComparisonImag;
    char* visibilitySourceFile;
    
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

typedef struct Visibility {
    FloatComplex comp;
    float u;
    float v;
    float w;
} Visibility;

typedef struct InterpolationPoint {
    float xShift;
    float yShift;
    DoubleComplex weight;
} InterpolationPoint;

/*--------------------------------------------------------------------
*   FUNCTION DEFINITIONS
*-------------------------------------------------------------------*/
void initConfig(void);
void initGridder(void);
void runGridder(void);
void checkShaderStatus(GLuint shader);
void checkProgramStatus(GLuint program);
void setShaderUniforms(void);
GLuint createShader(GLenum shaderType, const char* shaderSource);
GLuint createProgram(GLuint fragmentShader, GLuint vertexShader);
void timerEvent(int value);
float timedifference_msec(struct timeval t0, struct timeval t1);
void printTimesAverage(struct timeval realStart, int processStart, char description[]);
void saveGridToFile(int support);
void loadVisibilitySamples(void);
void compareToIdealGrid(void);
void createWProjectionPlanes(FloatComplex *wTextures);
void createPhaseScreen(int resolutionFullSupport, DoubleComplex *screen, double w, double fieldOfView, int wFullSupport);
double calcSpheroidalWeight(double nu);
void inverseFFT2dVectorRadixTransform(int numChannels, DoubleComplex *input, DoubleComplex *output);
void calcBitReversedIndices(int n, int* indices);
void fft2dShift(int n, DoubleComplex *input, DoubleComplex *shifted);

float calcResolutionShift(float index, float width);
float calcInterpolateShift(float index, float width);
double getShift(double width);
void getBicubicNeighbours(float xShift, float yShift, InterpolationPoint *neighbours, int resolutionSupport, DoubleComplex* matrix);
InterpolationPoint interpolateCubicWeight(InterpolationPoint *points, InterpolationPoint newPoint, int start, int width, bool horizontal);
void saveKernelToFile(char* filename, float w, int support, DoubleComplex* data);
void saveRadialKernelsToFile(char* filename, int support, int wPlanes, FloatComplex* data);
float calcSpheroidalShift(int index, int width);
int calcPosition(float x, int scalerWidth);
float calcShift(int index, int width, float start);

DoubleComplex complexAdd(DoubleComplex x, DoubleComplex y);
DoubleComplex complexSubtract(DoubleComplex x, DoubleComplex y);
DoubleComplex complexMultiply(DoubleComplex x, DoubleComplex y);
DoubleComplex complexConjugateExp(double ph);
double complexMagnitude(double x, double y);

void compareGrids(GLfloat *gridA, GLfloat *gridB, int gridDimension);
void compareGridsL2Norm(GLfloat *gridA, GLfloat *gridB, int gridDimension);
void compareGridsAnthonyNorm(GLfloat *gridA, GLfloat *gridB, int gridDimension);
void loadGridFromFile(GLfloat *grid, int gridDimension);
void generateHistogramFile(GLfloat *gridA, GLfloat *gridB, int width);
void cleanup(void);

#endif /* GRIDDER_H */

