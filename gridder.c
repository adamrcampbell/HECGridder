
/*
 * 
 * Authors: Seth Hall, Andrew Ensor, Adam Campbell
 * Auckland University of Technology - AUT
 * High Performance Laboratory
 * 
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <sys/time.h>
#include <string.h>
#include <unistd.h>

#include <stdbool.h>
#include <GL/glew.h>
#include <GL/freeglut.h>    

#include "gridder.h"
#include "gpu.h"

#ifndef M_PI
    #define M_PI 3.14159265358979323846264338327
#endif

// NAG Dataset configurations
//
//  File:           el82-70.txt
//  Grid Size:      18000
//  Min Support:    4.0
//  Max Support:    44.0
//  Max W:          7083.386050 // we assume
//  Max Plane:      339.0
//  Cell Size Rad:  0.000006
//
//  File:           el56-82.txt
//  Grid Size:      18000
//  Min Support:    4
//  Max Support:    72
//  Max W:          12534.770126 // we assume
//  Max Plane:      601
//  Cell Size Rad:  0.000006
//
//  File:           el30-56.txt
//  Grid Size:      18000
//  Min Support:    4
//  Max Support:    95
//  Max W:          19225.322282 // we assume
//  Max Plane:      922
//  Cell Size Rad:  0.000006
//
//  File:           synthetic.txt
//  Grid Size:      10000
//  Min Support:    4
//  Max Support:    36
//  Max W:          40920.395944 // we assume
//  Max Plane:      714
//  Cell Size Rad:  0.000004

/*--------------------------------------------------------------------
 *   GUI CONFIG
 *-------------------------------------------------------------------*/
static GLfloat guiRenderBounds[8];
static GLuint sProgram;
static GLuint sLocPosition;
static GLuint sComplex;
static GLuint guiRenderBoundsBuffer;
static GLuint visibilityBuffer;
static GLuint sLocPositionRender;
static GLuint sProgramRender;
static GLuint uShaderTextureHandle;
static GLuint uShaderTextureKernalHandle; 
static GLuint uMinSupportOffset;
static GLuint uWToMaxSupportRatio;
static GLuint uGridCenter;
static GLuint uGridCenterOffset;
static GLuint uGridSizeRender;
static GLuint uWScale;
static GLuint uWStep;
static GLuint uUVScale; 
static GLuint uNumPlanes;
static GLuint fboID;
static GLuint textureID;
static GLuint kernalTextureID;
static GLuint textureID;
static GLfloat* gridBuffer;
static GLenum KERNEL_DIM;
static FloatComplex* kernelBuffer;
static GLuint* visibilityIndices;
static GLfloat* visibilities;

// Used for counting gridding iterations performed
int iterationCount = 0;
int totalDumpsPerformed = 0;
int teminationDumpCount;

// Used for timing of gridder
int counter = 0;
int counterAverage;
double sumTimeReal;
float sumTimeProcess;
int val;
struct timeval timeCallsReal;
int timeCallsProcess;
bool toggle = 0;  
int windowDisplay;

// Global gridder configuration
Config config;


//CHANGES: - numWPlanes set to 5, gridDimension set to 100, changed texture to a 2D
//altered GPU.h fragment shader to allow Radial including use of Sampler2D, testing 1 visibility only
//kernel function now calling new test routine where calloc is called

void initConfig(void) 
{
    // Scale grid dimension down for GUI rendering
    windowDisplay = 900;
    
    // Full support texture dimension (must be power of 2 greater or equal to kernelMaxFullSupport)
    // Tradeoff note: higher values result in better precision, but result in more memory used and 
    // slower rendering to the grid in GPU.. NOTE RADIAL MODE USES ONLY HALF THIS VALUE
    config.kernelTexSize = 128;
    
    // Full support kernel resolution used for creating w projection kernels (always power of 2 greater than kernelTexSize)
    // Tradeoff note: higher values result in better precision, but result in a slower kernel creation for each plane
    // due to use of FFT procedure (512 is a good value to use)
    config.kernelResolutionSize = 512;
    
    // Single dimension of the grid
    config.gridDimension = 18000.0f;
    
    // Full support of min/max kernel supported per observation
    // Note: kernelMaxFullSupport must be less than or equal to kernelResolutionSize
    config.kernelMaxFullSupport = (44.0f * 2.0f) + 1.0f;
    config.kernelMinFullSupport = (4.0f * 2.0f) + 1.0f;
    
    // Number of visibilities to process (is set when reading visibilities from file)
    // Note: if not reading from file, then must be manually changed.
    config.visibilityCount = 1;
    
    // Flag to determine if reading visibilities from a source file
    config.visibilitiesFromFile = true;
    
    // Source of visibility data
    config.visibilitySourceFile = "datasets/el82-70.txt";
    
    // Scalar value for scaling visibility UVW wavelengths to coordinates
    config.frequencyStartHz = 100000000.0;
    
    // Flag to determine grid center offset (true: indicates grid points land in the middle of a pixel 
    // (same as oxford gridder), false: indicates grid points should fall in between pixels (other implementations))
    config.offsetVisibilities = true;
    
    // Uses heavy interpolation for convolving visibilities to grid (on GPU)
    // Note: slows down gridding a batch of visibilities, but improves precision
    config.useHeavyInterpolation = true;
    
    //used to enable radial Texturing (2D kernels) or Cube texturing (3D)
    config.useRadial = false;
    
    // Number of visibility attributes (U, V, W, Real, Imaginary, Weight) - does not change
    config.numVisibilityParams = 6;
    
    // Number of gridding iterations to perform before terminating (all visibilities convolved each iteration)
    config.displayDumpTime = 1;
    
    // variable used to control when the Gridder will exit after reaching the dump count, 
    // use a negative value to keep "infinite" gridding. 
    // Note: number of actual iterations is terminationDumpCount * displayDumpTime assuming dumpCount positive
    teminationDumpCount = -1;
    
    // Flag if want to compare HEC gridder output to Oxford gridder output (ensure file input locations are defined)
    // Note: only compares on first iteration, remainder are just processed for timing output and GUI rendering.
    // Also only can compare the two grids at the same interval as the dump time
    config.compareToOxfordGrid = true;
    
    // Source of Oxford grid output (real component)
    config.inputGridComparisonReal = "grids/oxford_grid_82-70_real.csv";
    
    // Source of Oxford grid output (imaginary component)
    config.inputGridComparisonImag = "grids/oxford_grid_82-70_imag.csv";
    
    // Used to slow down GUI rendering (milliseconds) - 0 means no delay, 1000 means one second delay
    config.refreshDelay = 0;
    
    // Calculates the OpenGL "world" coordinate system for gridding
    // Note: grid center (0.0, 0.0) is center of world
    float gridDimFloat = (float) config.gridDimension;
    GLfloat renderTemp[8] = {
        -gridDimFloat/2.0f, -gridDimFloat/2.0f,
        -gridDimFloat/2.0f, gridDimFloat/2.0f,
        gridDimFloat/2.0f, -gridDimFloat/2.0f,
        gridDimFloat/2.0f, gridDimFloat/2.0f
    };
    memcpy(guiRenderBounds, renderTemp, sizeof (guiRenderBounds));
    
    // Maximum W term to support
    config.wProjectionMaxW = 7083.386050;
    
    // Cell size radians for observation
    config.cellSizeRad = 0.00000639708380288949;
    
    // Number of W planes to create
    config.wProjectNumPlanes = 339;
    
    // Scales W terms (used on GPU to determine w plane index)
    config.wScale = pow((double) config.wProjectNumPlanes, 2.0) / config.wProjectionMaxW;
    
    // Field of view for observation (relies on original grid dimension)
    config.fieldOfView =  config.cellSizeRad * (double) config.gridDimension;
    
    // Custom variable for scaling the UVScale (used for testing, zooms the GUI rendered visibilities)
    config.graphicMultiplier = 1.0f;
    
    // Scales visibility UV coordinates to grid coordinates
    config.uvScale = (double) config.gridDimension * config.cellSizeRad * config.graphicMultiplier; 
    
    // Used to calculate required W full support per w term
    config.wToMaxSupportRatio = ((config.kernelMaxFullSupport - config.kernelMinFullSupport) / config.wProjectionMaxW);
}

int main(int argc, char** argv) {
    
    initConfig();
    
    srand((unsigned int) time(NULL));
    setenv("DISPLAY", ":0", 11.0);
    glutInit(&argc, argv);
    glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE | GLUT_ALPHA);
    glutInitWindowSize(windowDisplay, windowDisplay);
    glutInitWindowPosition(0, 0);
    glutCreateWindow("HEC Gridder");
    glutDisplayFunc(runGridder);
    glutTimerFunc(config.refreshDelay, timerEvent, 0);
    //glewInit();
    glewExperimental = GL_TRUE;
    GLenum err = glewInit();
    if (err != GLEW_OK) {
        printf("ERR: Unable to load and run Gridder!\n");
    }

    initGridder();
    glutMainLoop();
    return (EXIT_SUCCESS);
}

/*
 * Function: initGridder 
 * --------------------
 *  Initialises the openGL context for gridding, loads visibilities from file, 
 *  sets up vertex/fragment shaders for gridding, binds grid memory to GPU,
 *  and creates w projection kernels for later use.
 *
 *  returns: nothing
 */
void initGridder(void) 
{    
    glClampColorARB(GL_CLAMP_VERTEX_COLOR_ARB, GL_FALSE);
    glClampColorARB(GL_CLAMP_READ_COLOR_ARB, GL_FALSE);
    glClampColorARB(GL_CLAMP_FRAGMENT_COLOR_ARB, GL_FALSE);

    srand(time(NULL));
    int size = 4 * config.gridDimension*config.gridDimension;
    gridBuffer = (GLfloat*) malloc(sizeof (GLfloat) * size);
    memset(gridBuffer, 0, size);

    if(config.visibilitiesFromFile)
    {
        FILE *file = fopen(config.visibilitySourceFile, "r");
        
        if(file != NULL)
        {
            int visCount = 0;
            fscanf(file, "%d\n", &visCount);
            config.visibilityCount = visCount;
            printf("READING %d number of visibilities from file \n",config.visibilityCount);
            visibilities = malloc(sizeof (GLfloat) * config.numVisibilityParams * config.visibilityCount);
            float temp_uu, temp_vv, temp_ww = 0.0f;
            float temp_real, temp_imag = 0.0f, temp_weight = 0.0f;
            double scale = config.frequencyStartHz / 299792458.0; // convert from wavelengths
            
            for(int i = 0; i < config.visibilityCount * config.numVisibilityParams; i+=config.numVisibilityParams)
            {
                fscanf(file, "%f %f %f %f %f %f\n", &temp_uu, &temp_vv, &temp_ww, &temp_real, &temp_imag, &temp_weight);
                
                visibilities[i] = (-temp_uu * scale); // right ascension
                visibilities[i + 1] = (temp_vv * scale);
                visibilities[i + 2] = temp_ww * scale;
                visibilities[i + 3] = temp_real;
                visibilities[i + 4] = temp_imag;
                visibilities[i + 5] = temp_weight;
            }
            
            fclose(file);
        }
        else
            printf("NO VISIBILITY FILE\n");
    }
    else
        visibilities = malloc(sizeof (GLfloat) * config.numVisibilityParams * config.visibilityCount);
    
    visibilityIndices = malloc(sizeof (GLuint) * config.visibilityCount);

    for (GLuint i = 0; i < config.visibilityCount; i++) {
        visibilityIndices[i] = i;
    }
    
    GLuint vertexShader;
    
    if(config.useHeavyInterpolation)
        vertexShader = createShader(GL_VERTEX_SHADER, VERTEX_SHADER);
    else
        vertexShader = createShader(GL_VERTEX_SHADER, VERTEX_SHADER_SNAP);
    
    GLuint fragmentShader;
    if(config.useRadial)
        fragmentShader = createShader(GL_FRAGMENT_SHADER, FRAGMENT_SHADER_RADIAL);
    else
        fragmentShader = createShader(GL_FRAGMENT_SHADER, FRAGMENT_SHADER);
    
    sProgram = createProgram(vertexShader, fragmentShader);
    sLocPosition = glGetAttribLocation(sProgram, "position");
    sComplex = glGetAttribLocation(sProgram, "complex");
    uShaderTextureKernalHandle = glGetUniformLocation(sProgram, "kernalTex");
    uMinSupportOffset = glGetUniformLocation(sProgram, "minSupportOffset");
    uWToMaxSupportRatio = glGetUniformLocation(sProgram, "wToMaxSupportRatio");
    uGridCenter = glGetUniformLocation(sProgram, "gridCenter");
    uGridCenterOffset = glGetUniformLocation(sProgram, "gridCenterOffset");
    uWScale = glGetUniformLocation(sProgram, "wScale");
    uWStep = glGetUniformLocation(sProgram, "wStep");
    uUVScale = glGetUniformLocation(sProgram, "uvScale");
    uNumPlanes = glGetUniformLocation(sProgram, "numPlanes");

    vertexShader = createShader(GL_VERTEX_SHADER, VERTEX_SHADER_RENDER);
    fragmentShader = createShader(GL_FRAGMENT_SHADER, FRAGMENT_SHADER_RENDER);
    sProgramRender = createProgram(vertexShader, fragmentShader);
    sLocPositionRender = glGetAttribLocation(sProgramRender, "position");
    uShaderTextureHandle = glGetUniformLocation(sProgramRender, "destTex");
    uGridSizeRender = glGetUniformLocation(sProgramRender, "gridSize");
    glGenBuffers(1, &guiRenderBoundsBuffer);
    glBindBuffer(GL_ARRAY_BUFFER, guiRenderBoundsBuffer);
    glBufferData(GL_ARRAY_BUFFER, sizeof(GLfloat) * 8, guiRenderBounds, GL_STATIC_DRAW);
    glBindBuffer(GL_ARRAY_BUFFER, 0);

    //Generate Texture for output.
    glGenBuffers(1, &visibilityBuffer);
    GLuint idArray[2];
    glGenTextures(2, idArray);
    textureID = idArray[0];
    glBindTexture(GL_TEXTURE_2D, textureID);
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glEnable(GL_TEXTURE_2D);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, config.gridDimension, config.gridDimension, 0, GL_RGBA, GL_FLOAT, gridBuffer);

    glGenFramebuffers(1, idArray);
    fboID = idArray[0];
    glBindFramebuffer(GL_FRAMEBUFFER, fboID);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, textureID, 0);
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
    glBindTexture(GL_TEXTURE_2D, 0);
    
    if(config.useRadial)
    {   printf("RADIAL MODE SET, KERNELTEX SIZE should be HALF OF RESOLUTION FOR BEST RESULTS ");
        config.kernelTexSize = config.kernelResolutionSize/2;
        kernelBuffer = calloc(config.kernelTexSize* config.wProjectNumPlanes, sizeof(FloatComplex));
        createWProjectionPlanes(kernelBuffer);
        //createRadialPlanesTest(kernelBuffer, config.kernelTexSize, config.wProjectNumPlanes);
        KERNEL_DIM = GL_TEXTURE_2D;
    }
    else
    {    kernelBuffer = calloc(config.kernelTexSize * config.kernelTexSize * config.wProjectNumPlanes, sizeof(FloatComplex));
         createWProjectionPlanes(kernelBuffer);
         KERNEL_DIM = GL_TEXTURE_3D;
    }
    
    
    //kernal TEXTURE
    kernalTextureID = idArray[1];
    glBindTexture(KERNEL_DIM, kernalTextureID);
    glTexParameterf(KERNEL_DIM, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameterf(KERNEL_DIM, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(KERNEL_DIM, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(KERNEL_DIM, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    if(!config.useRadial)
        glTexParameteri(KERNEL_DIM, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE);
    glEnable(KERNEL_DIM);
    // width, height, 
    if(config.useRadial)
        glTexImage2D(GL_TEXTURE_2D, 0,  GL_RG32F, config.kernelTexSize, (int) config.wProjectNumPlanes, 0, GL_RG, GL_FLOAT, kernelBuffer);
    else
        glTexImage3D(GL_TEXTURE_3D, 0,  GL_RG32F, config.kernelTexSize, config.kernelTexSize, (int) config.wProjectNumPlanes, 0, GL_RG, GL_FLOAT, kernelBuffer);

    glBindTexture(KERNEL_DIM, 0);
    
    glTexEnvi(GL_POINT_SPRITE, GL_COORD_REPLACE, GL_TRUE);
    glEnable(GL_POINT_SPRITE);
    glEnable(GL_VERTEX_PROGRAM_POINT_SIZE);
    
    setShaderUniforms();
    glFlush();
    glFinish();
    counter = 0;
    sumTimeProcess = 0.0f;
    gettimeofday(&timeCallsReal, 0);
    timeCallsProcess = clock();
    
    printf("SEEMS LIKE ITS ALL SET UP FINE??? \n");
}

void createRadialPlanesTest(FloatComplex *kernel,int totalWidth, int totalHeight)
{
    float inc = 1.0f/(float)totalWidth;
    float inc2 = 1.0f/(float)totalHeight;
    
    for(int r = 0; r < totalHeight; r++)
    {   int c = 0;
        for(int c = 0; c < totalWidth-1; c++)
        {
            kernel[r * totalWidth + c].real = 1.0f - c*inc;   
            kernel[r * totalWidth + c].imaginary = 1.0 - r*inc2;
            if(r == 0) 
                printf("(%.3f, %d) ",  kernel[r * totalWidth + c].real, r * totalWidth + c);
        }
        //ensure last value is always 0 for radial values > 1 (ensures Clamps to 0)
        kernel[r * totalWidth + c].real = 0.0f;
        kernel[r * totalWidth + c].imaginary = 0.0f;
        //printf("\n");
    }
}

/*
 * Function: setShaderUniforms 
 * --------------------
 *  Sets gridding uniforms for use within vertex/fragment shaders
 *
 *  returns: nothing
 */
void setShaderUniforms(void)
{
    printf("SETTING THE SHADER UNIFORMS\n");
    glUseProgram(sProgram);
    glUniform1f(uMinSupportOffset, config.kernelMinFullSupport);
    glUniform1f(uWToMaxSupportRatio, (config.kernelMaxFullSupport-config.kernelMinFullSupport)/config.wProjectionMaxW); //(maxSuppor-minSupport) / maxW
    glUniform1f(uGridCenter, ((float) config.gridDimension - 0.5) / 2.0);
    float centerOffset = config.offsetVisibilities ? (0.5 / ((float) config.gridDimension / 2.0)) : 0.0;
    printf("Center Offset: %f\n", centerOffset);
    glUniform1f(uGridCenterOffset, centerOffset);
    glUniform1f(uWScale, config.wScale);
    glUniform1f(uWStep, 1.0 / (float) config.wProjectNumPlanes);
    glUniform1f(uUVScale, config.uvScale);
    glUniform1f(uNumPlanes, config.wProjectNumPlanes);
    glUseProgram(0);
    
    glUseProgram(sProgramRender);
    glUniform1f(uGridSizeRender, config.gridDimension);
    glUseProgram(0);  
    printf("DONE WITH SETTING THE SHADER UNIFORMS\n");
}

/*
 * Function: runGridder 
 * --------------------
 *  Binds visibility data to the GPU, grids visibility data on GPU,
 *  optionally renders grid to GUI, and optionally compares gridded result
 *  to another grid from file
 *
 *  returns: nothing
 */
void runGridder(void) {
    
    // Used for testing 
    if(!config.visibilitiesFromFile)
    {
        // convert from wavelengths
        double scale = config.frequencyStartHz / 299792458.0;
        
        for (int i = 0; i < config.visibilityCount * config.numVisibilityParams; i += config.numVisibilityParams) {
            // U, V, W, Real, Imaginary, Weight
            visibilities[i] =  0.0;// * scale;
            visibilities[i + 1] = 0.0;// * scale;
            visibilities[i + 2] = 7041.0f; //* scale;
            visibilities[i + 3] = 1.0;
            visibilities[i + 4] = 1.0;
            visibilities[i + 5] = 1.0f;
        }
    }

    glBindFramebuffer(GL_FRAMEBUFFER, fboID);
    glEnable(GL_BLEND);
    glViewport(0, 0, config.gridDimension, config.gridDimension);

    struct timeval timeFunctionReal;
    gettimeofday(&timeFunctionReal, 0);
    int timeFunctionProcess = clock();
    glUseProgram(sProgram);
    glBlendEquationSeparate(GL_FUNC_ADD, GL_FUNC_ADD);
    glBlendFuncSeparate(GL_ONE, GL_ONE, GL_ONE, GL_ONE);
    glBindTexture(KERNEL_DIM, kernalTextureID);
    glUniform1i(uShaderTextureKernalHandle, 0);
    glBindBuffer(GL_ARRAY_BUFFER, visibilityBuffer);
    glBufferData(GL_ARRAY_BUFFER, sizeof (GLfloat) * config.numVisibilityParams * config.visibilityCount, visibilities, GL_STATIC_DRAW);
    glEnableVertexAttribArray(sLocPosition);
    glVertexAttribPointer(sLocPosition, 3, GL_FLOAT, GL_FALSE, config.numVisibilityParams*sizeof(GLfloat), 0);
    glEnableVertexAttribArray(sComplex);
    glVertexAttribPointer(sComplex, 3, GL_FLOAT, GL_FALSE, config.numVisibilityParams*sizeof(GLfloat), (void*) (3*sizeof(GLfloat)));
    
    int batchSize = config.visibilityCount;
    int end = batchSize;
    int start = 0;
    //for(int i=0;i<10;i++)
    {   
        printf("doing batches from %d to %d\n", start, end);
       // glDrawElements(GL_POINT, 10, GL_UNSIGNED_INT, visibilityIndices);
        glDrawArrays(GL_POINTS, start, end);
        glFinish();
        
        start += batchSize;
        end +=batchSize;
    }
     
    for (GLenum err = glGetError(); err != GL_NO_ERROR; err = glGetError()) {
        fprintf(stderr, "%d: %s\n", err, gluErrorString(err));
    }
    glDisableVertexAttribArray(sComplex);
    glDisableVertexAttribArray(sLocPosition);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glBindTexture(KERNEL_DIM, 0);
    glUseProgram(0);
    
    glDisable(GL_BLEND);
    glBindFramebuffer(GL_FRAMEBUFFER, 0);

    glFinish();
    //DRAW RENDERING
    glViewport(0, 0, windowDisplay, windowDisplay);
    glUseProgram(sProgramRender);
    glBindTexture(GL_TEXTURE_2D, textureID);
    glUniform1i(uShaderTextureHandle, 0);

    glBindBuffer(GL_ARRAY_BUFFER, guiRenderBoundsBuffer);
    glEnableVertexAttribArray(sLocPositionRender);
    glVertexAttribPointer(sLocPositionRender, 2, GL_FLOAT, GL_FALSE, 0, 0);
    glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);

    for (GLenum err = glGetError(); err != GL_NO_ERROR; err = glGetError()) {
        fprintf(stderr, "%d: %s\n", err, gluErrorString(err));
    }
    glDisableVertexAttribArray(sLocPositionRender);
    glBindTexture(GL_TEXTURE_2D, 0);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glUseProgram(0);

    counter++;
    iterationCount++;
    
    bool dumped = false;
    if(iterationCount == config.displayDumpTime)
    {   printf("Dumping grid from GPU back to host\n");
        glBindFramebuffer(GL_FRAMEBUFFER, fboID);
        iterationCount = 0;
        
        // Ensure OpenGL has finished
        glFinish();
        glReadPixels(0, 0, config.gridDimension, config.gridDimension,  GL_RGBA, GL_FLOAT, gridBuffer);
        glFinish();
        
        // This function can be used if you wish to save the gridder results to file
        // Saves convolutional weights, grid real, and grid imaginary
       // saveGridToFile(config.gridDimension);
        
        glClearColor(0.0f, 0.0f, 0.0f, 0.0f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        glBindFramebuffer(GL_FRAMEBUFFER, 0);
        dumped = true;
        totalDumpsPerformed++;
    }
    
    glFinish();
    glutSwapBuffers();
    printTimesAverage(timeFunctionReal,timeFunctionProcess,"ENTIRE FUNCTION TIME");
    gettimeofday(&timeCallsReal, 0);
    timeCallsProcess = clock();
    
    if(config.compareToOxfordGrid && totalDumpsPerformed == 1 && dumped)
    {
        int gridElements = 4 * config.gridDimension * config.gridDimension;
        GLfloat *inputGrid = (GLfloat*) malloc(sizeof (GLfloat) * gridElements);
        loadGridFromFile(inputGrid, config.gridDimension);
        compareGrids(gridBuffer, inputGrid, config.gridDimension);
        generateHistogramFile(gridBuffer, inputGrid, config.gridDimension);
        free(inputGrid);
    }
    // Terminate program
    if(totalDumpsPerformed == teminationDumpCount)
        exit(0);
}

void timerEvent(int value) {
    glutPostRedisplay();
    glutTimerFunc(config.refreshDelay, timerEvent, 0);
}

float timedifference_msec(struct timeval t0, struct timeval t1)
{
    return (t1.tv_sec - t0.tv_sec) * 1000.0f + (t1.tv_usec - t0.tv_usec) / 1000.0f;
}

float msecAvg = 0;
float overallTimeAvg = 0;
float overallTimeSquareAvg = 0;
float timetimeAvg = 0;
float timetimeSquareAvg = 0;
int totalCount = 0;

void printTimesAverage(struct timeval realStart, int processStart, char description[])
{
    int timeTaken =  clock()-processStart;
    struct timeval realEnd;// = time(0);
    gettimeofday(&realEnd, 0);
    
    float msec = timeTaken * 1000.0f / (float)CLOCKS_PER_SEC;
    float timetime = timedifference_msec(realStart,realEnd);
    
    msecAvg+=msec;
    timetimeAvg +=timetime;
    overallTimeAvg += timetime;
    timetimeSquareAvg += (timetime*timetime);
    overallTimeSquareAvg += (timetime*timetime);
    if(counter == 1)
    {    
        totalCount += counter;
        float stdDev = (float)sqrt((timetimeSquareAvg/counter)-pow(timetimeAvg/counter,2.0));
        float stdDevOverall = (float)sqrt((overallTimeSquareAvg/totalCount)-pow(overallTimeAvg/totalCount,2.0));
        printf("%d> %s :\t Real time Avg = %.3f (%.3f STD), OVERALL time of = %.3f (%.3f STD)\n",
                totalCount,description,timetimeAvg/counter,stdDev,overallTimeAvg/totalCount,stdDevOverall);
        msecAvg=0;
        timetimeAvg =0;
        timetimeSquareAvg = 0;
        counter = 0; 
    }
}

/*
 * Function: checkShaderStatus 
 * --------------------
 *  Ensures an openGL shader is ready for use.
 *
 *  shader: The shader object to be verified
 * 
 *  returns: nothing
 */
void checkShaderStatus(GLuint shader) {
    GLint status;
    glGetShaderiv(shader, GL_COMPILE_STATUS, &status);
    printf("CHECKING SHADER STATUS\n");
    if (GL_FALSE == status) {
        GLint logLength;
        glGetShaderiv(shader, GL_INFO_LOG_LENGTH, &logLength);
        char *infoLog = malloc(logLength * sizeof (char));
        glGetShaderInfoLog(shader, logLength, NULL, infoLog);
        fprintf(stderr, "%d: %d, %s\n", __LINE__, GL_COMPILE_STATUS, infoLog);
        free(infoLog);
    }
    printf("DONE\n\n");
}

/*
 * Function: checkProgramStatus 
 * --------------------
 *  Ensures an openGL program is ready for use.
 *
 *  shader: The program object to be verified
 * 
 *  returns: nothing
 */
void checkProgramStatus(GLuint program) {
    GLint status;
    printf("CHECKING PROGRAM STATUS\n");
    glGetProgramiv(program, GL_LINK_STATUS, &status);
    if (GL_FALSE == status) {
        GLint logLength;
        glGetProgramiv(program, GL_INFO_LOG_LENGTH, &logLength);
        char *infoLog = malloc(logLength * sizeof (char));
        glGetProgramInfoLog(program, logLength, NULL, infoLog);
        fprintf(stderr, "%d: %d, %s\n", __LINE__, GL_LINK_STATUS, infoLog);
        free(infoLog);
    }
    printf("DONE\n\n");
}

/*
 * Function: createShader 
 * --------------------
 *  Creates a new instance of an OpenGL shader object
 *
 *  shaderType: The desired OpenGL shader type (eg; GL_VERTEX_SHADER)
 *  shaderSource: The location of the shader code (refer to gpu.h)
 * 
 *  returns: The new shader instance
 */
GLuint createShader(GLenum shaderType, const char* shaderSource) {
    GLuint shader = glCreateShader(shaderType);
    glShaderSource(shader, 1, (const GLchar **) &shaderSource, NULL);
    glCompileShader(shader);
    checkShaderStatus(shader);
    return shader;
}

/*
 * Function: createProgram 
 * --------------------
 *  Creates a new instance of an OpenGL program object
 *
 *  fragmentShader: The fragment shader stage for use within the program object
 *  vertexShader: The vertex shader stage for use within the program object
 * 
 *  returns: The new program instance
 */
GLuint createProgram(GLuint fragmentShader, GLuint vertexShader) {
    GLuint program = glCreateProgram();
    glAttachShader(program, fragmentShader);
    glAttachShader(program, vertexShader);
    glLinkProgram(program);
    checkProgramStatus(program);
    return program;
}

/*
 * Function: complexAdd 
 * --------------------
 *  Produces the sum two complex numbers
 *
 *  x : the first complex number
 *  y : the second complex number
 * 
 *  returns: The sum of the two complex numbers
 */
DoubleComplex complexAdd(DoubleComplex x, DoubleComplex y)
{
    DoubleComplex z;
    z.real = x.real + y.real;
    z.imaginary = x.imaginary + y.imaginary;
    return z;
}

/*
 * Function: complexSubtract 
 * --------------------
 *  Produces the difference of two complex numbers
 *
 *  x : the first complex number
 *  y : the second complex number
 * 
 *  returns: The difference of the two complex numbers
 */
DoubleComplex complexSubtract(DoubleComplex x, DoubleComplex y)
{
    DoubleComplex z;
    z.real = x.real - y.real;
    z.imaginary = x.imaginary - y.imaginary;
    return z;
}

/*
 * Function: complexMultiply 
 * --------------------
 *  Produces the product of two complex numbers
 *
 *  x : the first complex number
 *  y : the second complex number
 * 
 *  returns: The product of the two complex numbers
 */
DoubleComplex complexMultiply(DoubleComplex x, DoubleComplex y)
{
    DoubleComplex z;
    z.real = x.real*y.real - x.imaginary*y.imaginary;
    z.imaginary = x.imaginary*y.real + x.real*y.imaginary;
    return z;
}

/*
 * Function: complexConjugateExp 
 * --------------------
 *  Produces a complex conjugate based on the supplied phase term 
 *
 *  ph : the phase term being evaluated
 * 
 *  returns: The complex conjugate of the phase
 */
DoubleComplex complexConjugateExp(double ph)
{
    return (DoubleComplex) {.real = cos((double)(2.0*M_PI*ph)), .imaginary = -sin((double)(2.0*M_PI*ph))};
}

/*
 * Function: calcWFullSupport 
 * --------------------
 *  Calculates the full w support required for a specific w term
 *  (even supports are rounded up to next odd to ensure a peak in the prolate spheroidal)
 *
 *  w : the w term being evaluated
 *  wToMaxSupportRatio: The ratio between min/max full kernel support and the maximum w per observation
 *  minSupport: The minimum full support used in an observation
 * 
 *  returns: An odd full support dependant on the w term provided
 */
int calcWFullSupport(double w, double wToMaxSupportRatio, double minSupport)
{
    // Calculates the full support width of a kernel for w term
    // Round up to next odd (ensures peak in prolate spheroidal calculation)
    int wSupport = (int) (fabs(wToMaxSupportRatio * w) + minSupport);
    return (wSupport % 2 == 0) ? wSupport+1 : wSupport;
}

/*
 * Function: normalizeKernel 
 * --------------------
 * Normalizes a complex w projection kernel scaled for the full support 
 * which it will become when shrunk down on the GPU.
 *
 * kernel : the complex w projection kernel
 * resolution : the width/height of the kernel
 * support : the full support which to be scaled by for use on GPU
 * 
 * returns: nothing
 */
void normalizeKernel(DoubleComplex *kernel, int resolution, int support)
{
    // Get sum of realSum
    double realSum = 0.0, imagSum = 0.0;
    int r, c;
    for(r = 0; r < resolution; r++)
    {
        for(c = 0; c < resolution; c++)
        {
            realSum += kernel[r * resolution + c].real;   
            imagSum += kernel[r * resolution + c].imaginary;
        }
    }
    
    double scaleFactor = pow((double)resolution/(double)support, 2.0) / realSum;

    // Normalize weights
    for(r = 0; r < resolution; r++)
        for(c = 0; c < resolution; c++)
        {
            kernel[r * resolution + c].real *= scaleFactor;
            kernel[r * resolution + c].imaginary *= scaleFactor;
        }
}

/*
 * Function: interpolateKernel 
 * --------------------
 * Performs a two dimensional bicubic interpolation on a w projection kernel
 * to shrink it down to the desired full support specified in textureSupport
 * 
 * source : the source w projection kernel to be interpolated upon
 * destination : the destination memory for storing interpolated kernel
 * resolutionSupport : the width/height of the source kernel
 * textureSupport : the width/height of the destination kernel
 * 
 * returns: nothing
 */
void interpolateKernel(DoubleComplex *source, DoubleComplex* dest, int resolutionSupport, int textureSupport)
{   
    // Perform bicubic interpolation
    InterpolationPoint neighbours[16];
    InterpolationPoint interpolated[4];
    float xShift, yShift;
    
    for(int y = 0; y < textureSupport; y++)
    {
        yShift = calcInterpolateShift((float) y, (float) (textureSupport-1));
        
        for(int x = 0; x < textureSupport; x++)
        {
            xShift = calcInterpolateShift((float) x, (float) (textureSupport-1));
            getBicubicNeighbours(xShift, yShift, neighbours, resolutionSupport, source);
            
            for(int i  = 0; i < 4; i++)
            {
                InterpolationPoint newPoint = (InterpolationPoint) {.xShift = xShift, .yShift = neighbours[(i*4)+1].yShift};
                newPoint = interpolateCubicWeight(neighbours, newPoint, i*4, resolutionSupport-1, true);
                interpolated[i] = newPoint;
            }
            
            InterpolationPoint final = (InterpolationPoint) {.xShift = xShift, .yShift = yShift};
            final = interpolateCubicWeight(interpolated, final, 0, resolutionSupport-1, false);
            int index = y * textureSupport + x;
            dest[index] = (DoubleComplex) {.real = final.weight.real, .imaginary = final.weight.imaginary};
        }
    }
}

/*
 * Function: createWProjectionPlanes 
 * --------------------
 * Creates all w projection planes between 0 and maximum W term provided in
 * the gridder config struct.
 * 
 * wTextures : the block of memory for storing N number of w projection kernels
 * 
 * returns: nothing
 */
void createWProjectionPlanes(FloatComplex *wTextures)
{                
    int convolutionSize = config.kernelResolutionSize;
    int textureSupport = config.kernelTexSize;
    
    int convHalf = convolutionSize/2;
    int numWPlanes = config.wProjectNumPlanes;
    double wScale = config.wScale;
    double fov = config.fieldOfView;
    
    // Flat two dimension array
    DoubleComplex *screen = calloc(convolutionSize * convolutionSize, sizeof(DoubleComplex));
    // Create w screen
    DoubleComplex *shift = calloc(convolutionSize * convolutionSize, sizeof(DoubleComplex));
    // Single dimension spheroidal
    double *spheroidal = calloc(convolutionSize, sizeof(double));
    
     // Test variable to output W plane creation steps
    // (phase screen, after fft, interpolated, normalized)
    int plane = -1;
    printf("Num W Planes: %d\n", numWPlanes);
    for(int iw = 0; iw < numWPlanes; iw++)
    {        
        // Calculate w term and w specific support size
        double w = iw * iw / wScale;
        double fresnel = w * ((0.5 * config.fieldOfView)*(0.5 * config.fieldOfView));
        int wFullSupport = calcWFullSupport(w, config.wToMaxSupportRatio, config.kernelMinFullSupport);
        printf("Creating W Plane: (%d) For w = %f, field of view = %f, " \
                "fresnel number = %f, full w support: %d texSupport : %d\n", iw, w, config.fieldOfView, fresnel, wFullSupport, textureSupport);
        
        // Calculate Prolate Spheroidal
        createScaledSpheroidal(spheroidal, wFullSupport, convHalf);
        
        // Create Phase Screen
        createPhaseScreen(convolutionSize, screen, spheroidal, w, fov, wFullSupport);
        
        if(iw == plane)
            saveKernelToFile("output/wproj_%f_phase_screen_%d.csv", w, convolutionSize, screen);
        
        // Perform shift and inverse FFT of Phase Screen
        fft2dShift(convolutionSize, screen, shift);
        inverseFFT2dVectorRadixTransform(convolutionSize, shift, screen);
        fft2dShift(convolutionSize, screen, shift);
        
        if(iw == plane)
            saveKernelToFile("output/wproj_%f_after_fft_%d.csv", w, convolutionSize, shift);
        
        // Interpolate w projection kernel down to texture support dimensions
         DoubleComplex *interpolated;
        if(config.useRadial)
        {   interpolated = calloc(convolutionSize * convolutionSize, sizeof(DoubleComplex));
            memcpy(interpolated,shift,convolutionSize * convolutionSize* sizeof(DoubleComplex));
            normalizeKernel(interpolated, convolutionSize, wFullSupport);
            
            if(iw == plane)
                saveKernelToFile("output/wproj_%f_normalized_%d.csv", w, convolutionSize, interpolated);
        }
        else
        {
            interpolated = calloc(textureSupport * textureSupport, sizeof(DoubleComplex));
            interpolateKernel(shift, interpolated, convolutionSize, textureSupport);
            if(iw == plane)
                saveKernelToFile("output/wproj_%f_after_interpolated_%d.csv", w, textureSupport, interpolated);
            // Normalize the kernel
            normalizeKernel(interpolated, textureSupport, wFullSupport);
            if(iw == plane)
                saveKernelToFile("output/wproj_%f_normalized_%d.csv", w, textureSupport, interpolated);  
        }
        // Bind interpolated kernel to texture matrix
        
        if(config.useRadial)
        {   int halfPoint = convolutionSize*(convHalf)+(convHalf);
            for(int i=0;i<convHalf;i++)
            {   DoubleComplex interpWeight = interpolated[halfPoint + i];
                FloatComplex weight = (FloatComplex) {.real = (float) interpWeight.real, .imaginary = (float) interpWeight.imaginary};
                int index = (iw * convHalf)  + i;
                wTextures[index] = weight;
                if(i==(convHalf-1))
                {   wTextures[index].real = 0.0f;
                    wTextures[index].imaginary = 0.0f;
                }     
            }
        }
        else
        {   for(int y = 0; y < textureSupport; y++)
            {
                for(int x = 0; x < textureSupport; x++)
                {
                    DoubleComplex interpWeight = interpolated[y * textureSupport + x];
                    FloatComplex weight = (FloatComplex) {.real = (float) interpWeight.real, .imaginary = (float) interpWeight.imaginary};
                    int index = (iw * textureSupport * textureSupport) + (y * textureSupport) + x;
                    wTextures[index] = weight;
                }
                
            }  
        } 
        free(interpolated);
        memset(screen, 0, convolutionSize * convolutionSize * sizeof(DoubleComplex));
        memset(shift, 0, convolutionSize * convolutionSize * sizeof(DoubleComplex));
    }
    if(config.useRadial && plane >= 0)
    {
        saveRadialKernelsToFile("output/wproj_%d_radial_%d.csv",convHalf,numWPlanes,wTextures);
    }
    free(spheroidal);
    free(screen);
    free(shift);
}

/*
 * Function: createScaledSpheroidal 
 * --------------------
 * Creates a zero padded Prolate Spheroidal curve of wFullSupport width
 * 
 * spheroidal : the block of memory for storing the padded prolate spheroidal
 * wFullSupport : the width of the desired prolate spheroidal curve
 * convHalf : half the width of the number of elements in spheroidal
 * 
 * returns: nothing
 */
void createScaledSpheroidal(double *spheroidal, int wFullSupport, int convHalf)
{
    int wHalfSupport = wFullSupport/2;
    int paddedWFullSupp = wFullSupport+2;
    double *nu = calloc(paddedWFullSupp, sizeof(double));
    double *tempSpheroidal = calloc(paddedWFullSupp, sizeof(double));
    // Calculate steps
    for(int i = 0; i < paddedWFullSupp; i++)
        nu[i] = fabs(calcSpheroidalShift(i, paddedWFullSupp));
        
    // Calculate curve from steps
    calcSpheroidalCurve(nu, tempSpheroidal, paddedWFullSupp);
    
//    printf(">>> Printing Spheroidal\n");
//    for(int i = 0; i < paddedWFullSupp; i++)
//        printf("%f\n", tempSpheroidal[i]);
//    printf(">>> Done\n");
    
    // Antialiasing?
    for(int i = 0; i < paddedWFullSupp; i++)
        tempSpheroidal[i] *= (1.0 - pow(nu[i], 2.0));
    
//    printf(">>> Printing Spheroidal\n");
//    for(int i = 0; i < paddedWFullSupp; i++)
//        printf("%f\n", tempSpheroidal[i]);
//    printf(">>> Done\n");   
    
    // Bind weights to middle
    for(int i = convHalf-wHalfSupport; i <= convHalf+wHalfSupport; i++)
        spheroidal[i] = tempSpheroidal[i-(convHalf-wHalfSupport)+1];    
    
    free(tempSpheroidal);
    free(nu);
}

/*
 * Function: createPhaseScreen 
 * --------------------
 * Creates a phase screen dependant on the width/height of the scalarSupport
 * 
 * convSize : the full width/height of screen
 * screen : the memory block used for storing the calculated phase screen
 * spheroidal : the zero padded Prolate Spheroidal
 * w : the w term used to modify the phase of the screen
 * fieldOfView : the field of view required for the phase screen
 * scalarSupport : the full support required for the specified w term
 * 
 * returns: nothing
 */
void createPhaseScreen(int convSize, DoubleComplex *screen, double* spheroidal, double w, double fieldOfView, int scalarSupport)
{        
    int convHalf = convSize/2;
    int scalarHalf = scalarSupport/2;
    int index = 0;
    double taper, taperY;
    double l, m;
    double lsq, rsq;
    double phase;
    
    for(int iy = 0; iy < scalarSupport; iy++)
    {
        l = (((double) iy-(scalarHalf)) / (double) scalarSupport) * fieldOfView;
        lsq = l*l;
        taperY = spheroidal[iy+(convHalf-scalarHalf)];
        phase = 0.0;
        
        for(int ix = 0; ix < scalarSupport; ix++)
        {
            m = (((double) ix-(scalarHalf)) / (double) scalarSupport) * fieldOfView;
            rsq = lsq+(m*m);
            taper = taperY * spheroidal[ix+(convHalf-scalarHalf)];
            index = (iy+(convHalf-scalarHalf)) * convSize + (ix+(convHalf-scalarHalf));
            
            if(rsq < 1.0)
            {
                phase = w * (1.0 - sqrt(1.0 - rsq));
                screen[index] = complexConjugateExp(phase);
            }
            
            if(rsq == 0.0)
                screen[index] = (DoubleComplex) {.real = 1.0, .imaginary = 0.0};
                
            screen[index].real *= taper;
            screen[index].imaginary *= taper;
        }
    }
}

/*
 * Function: inverseFFT2dVectorRadixTransform 
 * --------------------
 * Performs an inverse FFT of the input memory block and stores it in 
 * the output memory block
 *  
 * numChannels : the number of channels to perform an iFFT on
 * input : the source data for performing an iFFT
 * output : the destination memory for the resulting iFFT
 * 
 * returns: nothing
 */
void inverseFFT2dVectorRadixTransform(int numChannels, DoubleComplex *input, DoubleComplex *output)
{   
    // Calculate bit reversed indices
    int* bitReversedIndices = malloc(sizeof(int) * numChannels);
    calcBitReversedIndices(numChannels, bitReversedIndices);
    
    // Copy data to result for processing
    for(int r = 0; r < numChannels; r++)
        for(int c = 0; c < numChannels; c++)
            output[r * numChannels + c] = input[bitReversedIndices[r] * numChannels + bitReversedIndices[c]];
    free(bitReversedIndices);
    
    // Use butterfly operations on result to find the DFT of original data
    for(int m = 2; m <= numChannels; m *= 2)
    {
        DoubleComplex omegaM = (DoubleComplex) {.real = cos(M_PI * 2.0 / m), .imaginary = sin(M_PI * 2.0 / m)};
        
        for(int k = 0; k < numChannels; k += m)
        {
            for(int l = 0; l < numChannels; l += m)
            {
                DoubleComplex x = (DoubleComplex) {.real = 1.0, .imaginary = 0.0};
                
                for(int i = 0; i < m / 2; i++)
                {
                    DoubleComplex y = (DoubleComplex) {.real = 1.0, .imaginary = 0.0};
                    
                    for(int j = 0; j < m / 2; j++)
                    {   
                        // Perform 2D butterfly operation in-place at (k+j, l+j)
                        int in00Index = (k+i) * numChannels + (l+j);
                        DoubleComplex in00 = output[in00Index];
                        int in01Index = (k+i) * numChannels + (l+j+m/2);
                        DoubleComplex in01 = complexMultiply(output[in01Index], y);
                        int in10Index = (k+i+m/2) * numChannels + (l+j);
                        DoubleComplex in10 = complexMultiply(output[in10Index], x);
                        int in11Index = (k+i+m/2) * numChannels + (l+j+m/2);
                        DoubleComplex in11 = complexMultiply(complexMultiply(output[in11Index], x), y);
                        
                        DoubleComplex temp00 = complexAdd(in00, in01);
                        DoubleComplex temp01 = complexSubtract(in00, in01);
                        DoubleComplex temp10 = complexAdd(in10, in11);
                        DoubleComplex temp11 = complexSubtract(in10, in11);
                        
                        output[in00Index] = complexAdd(temp00, temp10);
                        output[in01Index] = complexAdd(temp01, temp11);
                        output[in10Index] = complexSubtract(temp00, temp10);
                        output[in11Index] = complexSubtract(temp01, temp11);
                        y = complexMultiply(y, omegaM);
                    }
                    x = complexMultiply(x, omegaM);
                }
            }
        }
    }
    
    for(int i = 0; i < numChannels; i++)
        for(int j = 0; j < numChannels; j++)
        {
            output[i * numChannels + j].real /= (numChannels * numChannels);
            output[i * numChannels + j].imaginary /= (numChannels * numChannels);
        }
}

/*
 * Function: calcBitReversedIndices 
 * --------------------
 * Calculates an array of bit reversed indices for use within the iFFT
 *  
 * n : the number of indices for which to bit reverse
 * indices : the resulting bit reversed indices
 * 
 * returns: nothing
 */
void calcBitReversedIndices(int n, int* indices)
{   
    for(int i = 0; i < n; i++)
    {
        // Calculate index r to which i will be moved
        unsigned int iPrime = i;
        int r = 0;
        for(int j = 1; j < n; j*=2)
        {
            int b = iPrime & 1;
            r = (r << 1) + b;
            iPrime = (iPrime >> 1);
        }
        indices[i] = r;
    }
}

/*
 * Function: calcSpheroidalCurve 
 * --------------------
 * Calculates a Prolate Spheroidal curve by approximation
 * (the Fred Schwab PS approximation technique)
 * 
 * nu : the input weights of the PS for transformation
 * curve : the memory block for storing the calculated PS
 * width : the width of the PS curve to produce
 * 
 * returns: nothing
 */
void calcSpheroidalCurve(double *nu, double *curve, int width)
{   
    double p[2][5] = {{8.203343e-2, -3.644705e-1, 6.278660e-1, -5.335581e-1, 2.312756e-1},
                     {4.028559e-3, -3.697768e-2, 1.021332e-1, -1.201436e-1, 6.412774e-2}};
    double q[2][3] = {{1.0000000e0, 8.212018e-1, 2.078043e-1},
                     {1.0000000e0, 9.599102e-1, 2.918724e-1}};
    
    int pNum = 5;
    int qNum = 3;
    
    int *part = calloc(width, sizeof(int));
    double *nuend = calloc(width, sizeof(double));
    double *delnusq = calloc(width, sizeof(double));
    double *top = calloc(width, sizeof(double));
    double *bottom = calloc(width, sizeof(double));
    
    for(int i = 0; i < width; i++)
    {
        if(nu[i] >= 0.0 && nu[i] <= 0.75)
            part[i] = 0;
        if(nu[i] > 0.75 && nu[i] < 1.0)
            part[i] = 1;
        
        if(nu[i] >= 0.0 && nu[i] <= 0.75)
            nuend[i] = 0.75;
        if(nu[i] > 0.75 && nu[i] < 1.0)
            nuend[i] = 1.0;
        
        delnusq[i] = (nu[i] * nu[i]) - (nuend[i] * nuend[i]);
    }
    
    for(int i = 0; i < width; i++)
    {
        top[i] = p[part[i]][0];
        bottom[i] = q[part[i]][0];
    }
    
    for(int i = 1; i < pNum; i++)
        for(int y = 0; y < width; y++)
            top[y] += (p[part[y]][i] * pow(delnusq[y], i)); 
    
    for(int i = 1; i < qNum; i++)
        for(int y = 0; y < width; y++)
            bottom[y] += (q[part[y]][i] * pow(delnusq[y], i));
    
    for(int i = 0; i < width; i++)
    {   
        if(bottom[i] > 0.0)
            curve[i] = top[i] / bottom[i];
        if(fabs(nu[i]) > 1.0)
            curve[i] = 0.0;
        
    }
    
    free(bottom);
    free(top);
    free(delnusq);
    free(nuend);
    free(part);
}

/*
 * Function: fft2dShift 
 * --------------------
 * Performs a shift of data for use within an FFT/iFFT
 * 
 * n : the width/height of the input and shifted memory blocks
 * input : the data for which to shift
 * shifted : the memory block for storing the shifted data
 * 
 * returns: nothing
 */
void fft2dShift(int n, DoubleComplex *input, DoubleComplex *shifted)
{
    int r = 0, c = 0;
    for(int i = -n/2; i < n/2; i++)
    {
        for(int j = -n/2; j < n/2; j++)
        {
            if(i >= 0 && j >= 0)
                shifted[r * n + c] = input[i * n +j];
            else if(i < 0 && j >=0)
                shifted[r * n + c] = input[(i+n)*n+j];
            else if(i >= 0 && j < 0)
                shifted[r * n + c] = input[i*n+j+n];
            else
                shifted[r * n + c] = input[(i+n)*n+j+n];
            
            c++;
        }
        r++;
        c = 0;
    }
}

/*
 * Function: interpolateCubicWeight 
 * --------------------
 * Performs a bicubic interpolation on 4 supplied points
 * 
 * points : the 4 points to interpolate against
 * newPoint : the new point produced by the bicubic interpolation
 * start : the starting index of the 4 neighbours to use in the interpolation
 * width : the width of the original block of memory being interpolated from
 * horizontal : a flag to determine if performing a vertical or horizontal interpolation
 * 
 * returns: the newly interpolated weight
 */
InterpolationPoint interpolateCubicWeight(InterpolationPoint *points, InterpolationPoint newPoint, int start, int width, bool horizontal)
{      
    double shiftCubed = pow(getShift(width), 3);

    DoubleComplex p0 = (DoubleComplex) {.real = (horizontal) ? points[start+0].xShift : points[start+0].yShift, .imaginary = 0.0};
    DoubleComplex p1 = (DoubleComplex) {.real = (horizontal) ? points[start+1].xShift : points[start+1].yShift, .imaginary = 0.0};
    DoubleComplex p2 = (DoubleComplex) {.real = (horizontal) ? points[start+2].xShift : points[start+2].yShift, .imaginary = 0.0};
    DoubleComplex p3 = (DoubleComplex) {.real = (horizontal) ? points[start+3].xShift : points[start+3].yShift, .imaginary = 0.0};
    DoubleComplex interpShift = (DoubleComplex) {.real = (horizontal) ? newPoint.xShift : newPoint.yShift, .imaginary = 0.0};
    
    DoubleComplex w0 = (DoubleComplex) {.real = -(points[start+0].weight.real) / (6.0 * shiftCubed), 
            .imaginary = -(points[start+0].weight.imaginary) / (6.0 * shiftCubed)};
    DoubleComplex w1 = (DoubleComplex) {.real = points[start+1].weight.real / (2.0 * shiftCubed),
            .imaginary = points[start+1].weight.imaginary / (2.0 * shiftCubed)};
    DoubleComplex w2 = (DoubleComplex) {.real = -points[start+2].weight.real / (2.0 * shiftCubed), 
            .imaginary = -points[start+2].weight.imaginary / (2.0 * shiftCubed)};
    DoubleComplex w3 = (DoubleComplex) {.real = points[start+3].weight.real / (6.0 * shiftCubed), 
            .imaginary = points[start+3].weight.imaginary / (6.0 * shiftCubed)}; 
    
    DoubleComplex t0 = complexMultiply(complexMultiply(complexMultiply(w0, complexSubtract(interpShift, p1)), complexSubtract(interpShift, p2)), 
            complexSubtract(interpShift, p3));
    DoubleComplex t1 = complexMultiply(complexMultiply(complexMultiply(w1, complexSubtract(interpShift, p0)), complexSubtract(interpShift, p2)), 
            complexSubtract(interpShift, p3));
    DoubleComplex t2 = complexMultiply(complexMultiply(complexMultiply(w2, complexSubtract(interpShift, p0)), complexSubtract(interpShift, p1)),
            complexSubtract(interpShift, p3));
    DoubleComplex t3 = complexMultiply(complexMultiply(complexMultiply(w3, complexSubtract(interpShift, p0)), complexSubtract(interpShift, p1)), 
            complexSubtract(interpShift, p2));
    
    newPoint.weight = complexAdd(complexAdd(complexAdd(t0, t1), t2), t3);
    return newPoint;
}

/*
 * Function: getBicubicNeighbours 
 * --------------------
 * Calculates and stores the 16 elements required to perform a series of bicubic interpolation against
 * 
 * xShift : the x shift position to begin selecting neighbours from (-1.0 to 1.0 inclusive)
 * yShift : the y shift position to begin selecting neighbours from (-1.0 to 1.0 inclusive)
 * neighbours : the memory block for storing the 16 weights located
 * resolutionSupport : the full support of the convolution kernel being used for interpolation
 * matrix : the convolution kernel  
 * 
 * returns: nothing
 */
void getBicubicNeighbours(float xShift, float yShift, InterpolationPoint *neighbours, int resolutionSupport, DoubleComplex* matrix)
{
    // Get x, y from scaled shift 
    int scaledPosX = calcPosition(xShift, resolutionSupport);
    int scaledPosY = calcPosition(yShift, resolutionSupport);

    // Get 16 neighbours
    for(int r = scaledPosY - 1, i = 0; r < scaledPosY + 3; r++)
    {
        for(int c = scaledPosX - 1; c < scaledPosX + 3; c++)
        {
            InterpolationPoint n = (InterpolationPoint) {.xShift = calcInterpolateShift(c-1, resolutionSupport-2),
                .yShift = calcInterpolateShift(r-1, resolutionSupport-2)};
            
            if(c < 1 || c >= resolutionSupport || r < 1 || r >= resolutionSupport)
            {
                n.weight = (DoubleComplex) {.real = 0.0, .imaginary = 0.0};   
            }
            else
            {
                n.weight = matrix[r * resolutionSupport + c];
            }
            
            neighbours[i++] = n;
        }
    }
}

float calcSpheroidalShift(int index, int width)
{   
    // Even
    if(width % 2 == 0)
        return -1.0 + index * getShift(width);
    // Odd
    else
        return -1.0 + index * getShift(width-1);
}

float calcInterpolateShift(float index, float width)
{
    return -1.0 + (index * (2.0 / width));
}

float calcShift(int index, int width, float start)
{
    return start + (index * getShift(width));
}

double getShift(double width)
{
    return 2.0/width;
}

float getStartShift(float width)
{
    return -1.0 + (1.0 / width);
}

int calcPosition(float x, int scalerWidth)
{
    int offset = (x < 0.0) ? 1 : 2;
    return ((int) floor(((x+1.0f)/2.0f) * (scalerWidth-offset)))+1;
}

void saveKernelToFile(char* filename, float w, int support, DoubleComplex* data)
{
    char *buffer[100];
    sprintf(buffer, filename, w, support);
    FILE *file = fopen(buffer, "w");
    for(int r = 0; r < support; r++)
    {
        for(int c = 0; c < support; c++)
        {
            fprintf(file, "%f, ", data[r * support + c].real);
        }
        fprintf(file, "\n");
    }
    fclose(file);
    printf("FILE SAVED\n");
}
void saveRadialKernelsToFile(char* filename, int support, int wPlanes, FloatComplex* data)
{
    char *buffer[100];
    sprintf(buffer, filename, wPlanes, support);
    FILE *file = fopen(buffer, "w");
    for(int r = 0; r < wPlanes; r++)
    {   
        int index = r * support; 
        for(int c = 0; c < support; c++)
        {
                fprintf(file, "%f, ", data[index+c].real);
        }
        fprintf(file, "\n");
    }
    fclose(file);
    printf("FILE SAVED\n");
}
void saveGridToFile(int support)
{    
    int saveSupport = config.gridDimension;             // 512
    int saveRowMin = (support/2)-(saveSupport/2);       // 9000 - 256
    int saveRowMax = (support/2)+(saveSupport/2);       // 9000 + 256
    int saveColMin = (support*4/2)-(saveSupport*4/2);
    int saveColMax = (support*4/2)+(saveSupport*4/2);
    
    // Save to file (real portion)
    FILE *file_real = fopen("output/grid_real.csv", "w");
    FILE *file_imag = fopen("output/grid_imag.csv", "w");
    FILE *file_weight = fopen("output/grid_weight.csv", "w");
    for(int r = saveRowMin; r < saveRowMax; r++)
    {
        for(int c = saveColMin; c < saveColMax; c+=4)
        {   fprintf(file_weight, "%+f, ", gridBuffer[(r*(int)config.gridDimension*4)+c]);
            fprintf(file_real, "%+f, ", gridBuffer[(r*(int)config.gridDimension*4)+c+1]);
            fprintf(file_imag, "%+f, ", gridBuffer[(r*(int)config.gridDimension*4)+c+2]);
        }
        fprintf(file_real, "\n");
        fprintf(file_imag, "\n");
        fprintf(file_weight, "\n");
    }
    fclose(file_real);
    fclose(file_imag);
    fclose(file_weight);
    printf("Grid has been output to file\n");
}

void loadGridFromFile(GLfloat *grid, int gridDimension)
{
    printf(">>> Reading input grid\n");
    
    FILE *realFile = fopen(config.inputGridComparisonReal, "r");
    FILE *imagFile = fopen(config.inputGridComparisonImag, "r");
    int index = 0;
    float real = 0.0, imaginary = 0.0;
    float rMin = 100000.0, rMax = 0.0, iMin = 100000.0, iMax = 0.0;
    
    for(int r = 0; r < gridDimension; r++)
    {
        for(int c = 0; c < gridDimension * 4; c+=4)
        {
            index = (r * gridDimension * 4) + c;
            fscanf(realFile, "%f, ", &real);
            fscanf(imagFile, "%f, ", &imaginary);
            grid[index+1] = real;
            grid[index+2] = imaginary;
            
            if(real < rMin)
                rMin = real;
            if(real > rMax)
                rMax = real;

            if(imaginary < iMin)
                iMin = imaginary;
            if(imaginary > iMax)
                iMax = imaginary;
        }
    }
    
    printf("rMin: %f, rMax: %f, iMin: %f, iMax: %f\n", rMin, rMax, iMin, iMax);
    
    fclose(realFile);
    fclose(imagFile);
     
    printf(">>> Reading input grid - COMPLETE\n");
}

void compareGrids(GLfloat *gridA, GLfloat *gridB, int gridDimension)
{
    printf(">>> Comparing grids\n");
    
    int counter = 0, index = 0;
    float sum = 0.0, maxDistance = 0.0, rDiff = 0.0, iDiff = 0.0, distance = 0.0;
    float rMin = 100000.0, rMax = 0.0, iMin = 100000.0, iMax = 0.0;
    float realSumHEC = 0.0, imagSumHEC = 0.0, realSumNAG = 0.0, imagSumNAG = 0.0;
    
    for(int r = 0; r < gridDimension; r++)
    {
        for(int c = 0; c < gridDimension*4; c+=4)
        {
            index = (r * gridDimension * 4) + c;
            
            // Accumulate sums
            realSumHEC += gridA[index+1];
            imagSumHEC += gridA[index+2];
            realSumNAG += gridB[index+1];
            imagSumNAG += gridB[index+2];
            
            if(gridA[index+1] < rMin)
                rMin = gridA[index+1];
            if(gridA[index+1] > rMax)
                rMax = gridA[index+1];

            if(gridA[index+2] < iMin)
                iMin = gridA[index+2];
            if(gridA[index+2] > iMax)
                iMax = gridA[index+2];
            
            
            if((fabs(gridA[index+1]) + fabs(gridA[index+2])) > 0.0)
            {
                rDiff = gridA[index+1] - gridB[index+1];
                iDiff = gridA[index+2] - gridB[index+2];
                
                distance = (rDiff * rDiff) + (iDiff * iDiff);
                
                if(distance > maxDistance)
                    maxDistance = distance;
                
                sum += distance;
                counter++;
            }
        }
    }
 
    printf("rMin: %f, rMax: %f, iMin: %f, iMax: %f\n", rMin, rMax, iMin, iMax);
    printf("Sum: %f, Counter: %d, Max Distance: %f\n", sum, counter, maxDistance);
    printf("Sum of difference (visibilities): %f\n", sum / config.visibilityCount);
    sum /= counter;
    printf("Sum of difference: %f\n\n", sum);
    printf("HEC Real: %f, HEC Imag: %f\n", realSumHEC, imagSumHEC);
    printf("NAG Real: %f, NAG Imag: %f\n", realSumNAG, imagSumNAG);
    printf("Diff Real: %f, Diff Imag: %f\n\n", fabsf(realSumHEC-realSumNAG), fabsf(imagSumHEC-imagSumNAG));
    
    printf(">>> Comparing grids - COMPLETE\n");
}

void generateHistogramFile(GLfloat *gridA, GLfloat *gridB, int width)
{
    printf(">>> Generating histogram data...\n");
    FILE *file = fopen("histogram_data.csv", "w");
    DoubleComplex elementA, elementB, difference;
    double magnitudeA = 0, magnitudeB = 0;
    int index = 0;
    
    for(int r = 0; r < width; r++)
    {
        for(int c = 0; c < width*4; c+=4)
        {
            index = (r * width * 4) + c;
            // Get element
            elementA = (DoubleComplex) {.real = gridA[index+1], .imaginary = gridA[index+2]};
            elementB = (DoubleComplex) {.real = gridB[index+1], .imaginary = gridB[index+2]};
            // Calculate magnitude
            magnitudeA = sqrt(elementA.real * elementA.real + elementA.imaginary * elementA.imaginary);
            magnitudeB = sqrt(elementB.real * elementB.real + elementB.imaginary * elementB.imaginary);
            
            difference = complexSubtract(elementA, elementB);
            double magnitudeDiff = sqrt(difference.real * difference.real + difference.imaginary * difference.imaginary);
            if(magnitudeDiff > 0.01)
                 fprintf(file, "%.10f, %.10f\n", difference.real, difference.imaginary);
            
        }
    }
    fclose(file);
    printf(">>> Histogram data saved to file\n");
}


