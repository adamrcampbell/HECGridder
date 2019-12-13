
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
enum FragShaderType {FullCube = 0, Radial = 1, Reflect = 2};
/*--------------------------------------------------------------------
*   STRUCTS
*-------------------------------------------------------------------*/
typedef struct Config {
    // General
    unsigned int gridDimension;
    unsigned int renderDimension;
    unsigned int imageSize;
    unsigned int kernelTexSize;
    unsigned int kernelResolutionSize;
    double kernelMaxFullSupport;
    double kernelMinFullSupport;
    unsigned int visibilityCount;
    unsigned int numVisibilityParams;
    bool visibilitiesFromFile;
    double frequencyStartHz;
    bool offsetVisibilities;
    bool useHeavyInterpolation;
    bool accumulateMode;
    bool saveGridToFile;
    unsigned int numVectorElements;

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
    unsigned int interpolateTextures;
    
} Config;

typedef struct FloatComplex {
    float real;
    float imaginary;
} FloatComplex;

typedef struct DoubleComplex {
    double real;
    double imaginary;
} DoubleComplex;

typedef struct Timer {
    double accumulatedTimeMS;
    double currentAvgTimeMS;
    double sumOfSquareDiffTimeMS;
    int iterations;
} Timer;
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

void inverseFFT2dVectorRadixTransform(int numChannels, DoubleComplex *input, 
        DoubleComplex *output);

void calcBitReversedIndices(int n, int* indices);

void fft2dShift(int n, DoubleComplex *input, DoubleComplex *shifted);

double calcInterpolateShift(double index, double width);

void saveKernelToFile(char* filename, double w, int support, DoubleComplex* data);

void saveRadialKernelsToFile(char* filename, int support, int wPlanes, 
        FloatComplex* data);

int calcRelativeIndex(double x, double width);

double calcSphrShift(double index, double width);

void interpolateKernel(DoubleComplex *source, DoubleComplex *destination, 
    int sourceSupport, int destinationSupport);

// void getBicubicNeighbours(double rowShift, double colShift, DoubleComplex *n, 
//         double *rs, double *cs, int sourceSupport, DoubleComplex *source,
//         const int oversampled_support);





void interpolate_kernel_tiny(DoubleComplex *screen, DoubleComplex *texture, 
    int screen_size, int texture_size, int oversampled_support);

void getBicubicNeighboursTiny(double rowShift, double colShift, DoubleComplex *n, double *rs, double *cs,
        int sourceSupport, DoubleComplex *source, int oversampled_support);






void getBicubicNeighbours(double rowShift, double colShift, DoubleComplex *n, 
        double *rs, double *cs, int sourceSupport, DoubleComplex *source);

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

double calcAndrewShift(int index, int fullSupport);

void saveGriddingStats(char *filename);

void createPhaseScreenNew(int iw, int full_support, int conv_size, 
    double sampling, double w_scale, DoubleComplex *screen);



    void create_w_projection_kernels(FloatComplex *w_textures);

    DoubleComplex complex_scale(const DoubleComplex z, const double scalar);
    
    void fft_shift_in_place(DoubleComplex *matrix, const int size);
    
    double calculate_window_stride(const int index, const int width);

    double prolate_spheroidal(double nu);
    
    void generate_phase_screen(const int iw, const int conv_size, const int inner,
        const double sampling, const double w_scale, double *taper, DoubleComplex *screen);
    
    void fft_2d(DoubleComplex *matrix, int number_channels);
    
    void calc_bit_reverse_indices(int n, int* indices);
    
    void interpolate_kernel(DoubleComplex *screen, DoubleComplex *texture, 
        const int screen_size, const int texture_size, const double support,
        const int oversample);
    
    double calculate_support(const double w, const int min_support, 
        const double w_max_support_ratio);
    
    void normalize_kernel(DoubleComplex *kernel, const int texture_size, const double support);

#endif /* GRIDDER_H */
