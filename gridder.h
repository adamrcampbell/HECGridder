
#ifndef GRIDDER_H
#define GRIDDER_H

/*--------------------------------------------------------------------
*   STRUCTS
*-------------------------------------------------------------------*/
typedef struct Config {
    // General
    float gridDimension;
    unsigned int kernelTexSize;
    unsigned int KernelResolutionSize;
    unsigned int kernelMaxFullSupport;
    unsigned int kernelMinFullSupport;
    unsigned int visibilityCount;
    unsigned int numVisibilityParams;
    bool visibilitiesFromFile;
    char* visibilitySourceFile;

    // GUI
    unsigned int refreshDelay;
    unsigned int displayDumpTime;
    
    // W Projection
    float cellSizeRad;
    float uvScale;
    float wScale;
    float fieldOfView;
    float wProjectionStep;
    float wProjectionMaxW;
    unsigned int wProjectNumPlanes;
    double wToMaxSupportRatio;
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
void saveGridToFile(void);
void loadVisibilitySamples(void);
void compareToIdealGrid(void);
void calculateSpheroidalCurve(float * nu, int kernelWidth);
void createWProjectionPlanes(int convolutionSize, int numWPlanes, int textureSupport, double wScale, double fov);
void createPhaseScreen(int convSize, DoubleComplex *screen, double* spheroidal, double w, double fieldOfView, int scalarSupport);
void calcSpheroidalCurve(double *nu, double *curve, int width);
void inverseFFT2dVectorRadixTransform(int numChannels, DoubleComplex *input, DoubleComplex *output);
void calcBitReversedIndices(int n, int* indices);
void fft2dShift(int n, DoubleComplex *input, DoubleComplex *shifted);
void printGrid(void);

DoubleComplex normalizeWeight(DoubleComplex weight, double mag, int resolution, int support);
float calcInterpolateShift(int index, int width, float start);
double getShift(double width);
void getBicubicNeighbours(int x, int y, InterpolationPoint *neighbours, int kernelFullSupport, int interpFullSupport, DoubleComplex* matrix);
InterpolationPoint interpolateCubicWeight(InterpolationPoint *points, InterpolationPoint newPoint, int start, int width, bool horizontal);
void createScaledSpheroidal(double *spheroidal, int wFullSupport, int convHalf);
void saveKernelToFile(char* filename, float w, int support, DoubleComplex* data);
float calcSpheroidalShift(int index, int width);
int calcPosition(float x, int scalerWidth);
float calcShift(int index, int width, float start);

DoubleComplex complexAdd(DoubleComplex x, DoubleComplex y);
DoubleComplex complexSubtract(DoubleComplex x, DoubleComplex y);
DoubleComplex complexMultiply(DoubleComplex x, DoubleComplex y);
DoubleComplex complexConjugateExp(double ph);
double complexMagnitude(DoubleComplex x);

#endif /* GRIDDER_H */
