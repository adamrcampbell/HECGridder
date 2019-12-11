
/*
 * 
 * Authors: Dr. Seth Hall, Dr. Andrew Ensor, Adam Campbell
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
#include "gridder.h"

#ifndef M_PI
    #define M_PI 3.14159265358979323846264338327
#endif

#ifndef SPEED_OF_LIGHT
    #define SPEED_OF_LIGHT 299792458.0
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
static GLuint uShaderTextureKernelHandle; 
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
static GLuint kernelTextureID;
static GLuint textureID;
static GLfloat* gridBuffer;
static GLenum KERNEL_DIM;
static FloatComplex* kernelBuffer;
static GLfloat* visibilities;

// Used for counting gridding iterations performed
int iterationCount = 0;
int totalDumpsPerformed = 0;
int teminationDumpCount;
int windowDisplay;
// Global gridder configuration
Config config;
bool COMPARE_TO_ANTHONY = false;
Timer timer;
char* timingOutputFile;

void initConfig(char** argv) 
{
    // Scale grid dimension down for GUI rendering
    windowDisplay = 1024;
    
    // Full support texture dimension (must be power of 2 greater or equal to kernelMaxFullSupport)
    // Tradeoff note: higher values result in better precision, but result in more memory used and 
    // slower rendering to the grid in GPU.. NOTE RADIAL MODE USES ONLY HALF THIS VALUE
    config.kernelTexSize = 32;

    // Full support kernel resolution used for creating w projection kernels (always power of 2 greater than kernelTexSize)
    // Tradeoff note: higher values result in better precision, but result in a slower kernel creation for each plane
    // due to use of FFT procedure (512 is a good value to use)
    config.kernelResolutionSize = 4096;
    
    // Single dimension of the grid
    config.gridDimension = 2458; //2048;
    config.renderDimension = 16;
    config.imageSize = 2458;
    
    
    // Full support of min/max kernel supported per observation
    // Note: kernelMaxFullSupport must be less than or equal to kernelResolutionSize
    config.kernelMinFullSupport = (4.0 * 2.0) + 1.0;
    config.kernelMaxFullSupport = (4.0 * 2.0) + 1.0;
    
    // Number of visibilities to process (is set when reading visibilities from file)
    // Note: if not reading from file, then must be manually changed.
    config.visibilityCount = 1;
    
    // Flag to determine if reading visibilities from a source file
    config.visibilitiesFromFile = false;
    
    // Source of visibility data
    config.visibilitySourceFile = "data/GLEAM_small_visibilities.csv"; //"datasets/el82-70.txt";
    
    // Scalar value for scaling visibility UVW wavelengths to coordinates
    config.frequencyStartHz = SPEED_OF_LIGHT; //1.0000e+08;
    
    // Flag to determine grid center offset (true: indicates grid points land in the middle of a pixel 
    // (same as oxford gridder), false: indicates grid points should fall in between pixels (other implementations))
    config.offsetVisibilities = true;
    
    // Uses heavy interpolation for convolving visibilities to grid (on GPU)
    // Note: slows down gridding a batch of visibilities, but improves precision
    config.useHeavyInterpolation = true;
    
    config.accumulateMode = false;
    
    config.numVectorElements = 2;
    
    // Flag to specify which fragment shader technique
    // to use when rendering frags
    // Options: FullCube, Reflect, Radial
    config.fragShaderType = Reflect;
    
    config.interpolateTextures = 1; // 0 = nearest, 1 = interpolate
    
    // Number of visibility attributes (U, V, W, Real, Imaginary, Weight) - does not change
    config.numVisibilityParams = 6;
    
    // Number of gridding iterations to perform before terminating (all visibilities convolved each iteration)
    config.displayDumpTime = 1;
    
    // variable used to control when the Gridder will exit after reaching the dump count, 
    // use a negative value to keep "infinite" gridding. 
    // Note: number of actual iterations is terminationDumpCount * displayDumpTime assuming dumpCount positive
    teminationDumpCount = 1;
    
    //flag to save resulting grid to file (does this at dump time)
    config.saveGridToFile = true;
    
    //Name output grids, ignored if above variable false;
    config.outputGridReal = "data/hec_grid_real.csv";
    config.outputGridImag = "data/hec_grid_imag.csv";
    
    // Used to slow down GUI rendering (milliseconds) - 0 means no delay, 1000 means one second delay
    config.refreshDelay = 0;
    
    GLfloat renderTemp[8] = {
        -1.0f, -1.0f,
        -1.0f,  1.0f,
         1.0f, -1.0f,
         1.0f,  1.0f
    };
    memcpy(guiRenderBounds, renderTemp, sizeof (guiRenderBounds));
    
    // MaximuFm W term to support
    config.wProjectionMaxW = 1895.410847844;
    
    // Cell size radians for observation
    config.cellSizeRad =  8.52211548825356E-06;
    
    // Number of W planes to create
    config.wProjectNumPlanes = 17;
    
    // Scales W terms (used on GPU to determine w plane index)
    config.wScale = pow((double) config.wProjectNumPlanes-1, 2.0) / config.wProjectionMaxW;
    
    // Field of view for observation (relies on original grid dimension)
    config.fieldOfView =  config.cellSizeRad * (double)  config.imageSize;
    
    // Scales visibility UV coordinates to grid coordinates
    config.uvScale = (double) config.gridDimension * config.cellSizeRad; 
    
    // Used to calculate required W full support per w term
    config.wToMaxSupportRatio = ((config.kernelMaxFullSupport - config.kernelMinFullSupport) / config.wProjectionMaxW);
    
    timer.accumulatedTimeMS     = 0.0;
    timer.currentAvgTimeMS      = 0.0;
    timer.sumOfSquareDiffTimeMS = 0.0;
    timer.iterations            = 0;
    timingOutputFile = argv[11];
}

int main(int argc, char** argv)
{
    initConfig(argv); 
    
    setenv("DISPLAY", ":0", 11.0);
    glutInit(&argc, argv);
    glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE | GLUT_ALPHA);
    glutInitWindowSize(windowDisplay, windowDisplay);
    glutInitWindowPosition(0, 0);
    glutCreateWindow("HEC Gridder");
    glutDisplayFunc(runGridder);
    glutTimerFunc(config.refreshDelay, timerEvent, 0);
    glewExperimental = GL_TRUE;
    GLenum err = glewInit();
    if (err != GLEW_OK) {
        printf("ERR: Unable to load and run Gridder!\n");
    }

    initGridder();
    glutMainLoop();

    return EXIT_SUCCESS;
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

    int size = config.numVectorElements * config.renderDimension*config.renderDimension;
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
            double scale = config.frequencyStartHz / SPEED_OF_LIGHT; // convert from wavelengths
            
            for(int i = 0; i < config.visibilityCount * config.numVisibilityParams; i+=config.numVisibilityParams)
            {
                if(COMPARE_TO_ANTHONY)
                {
                    fscanf(file, "%f,%f,%f,%f\n", &temp_uu, &temp_vv,&temp_real, &temp_imag);
                    visibilities[i] = (-temp_uu * scale); // right ascension
                    visibilities[i + 1] = (temp_vv * scale);
                    visibilities[i + 2] = temp_ww;
                    visibilities[i + 3] = temp_real;
                    visibilities[i + 4] = temp_imag;
                    visibilities[i + 5] = 1.0f;
                }
                else
                {
                    fscanf(file, "%f %f %f %f %f %f\n", &temp_uu, &temp_vv, &temp_ww, &temp_real, &temp_imag, &temp_weight);
                    visibilities[i] = (-temp_uu * scale); // right ascension
                    visibilities[i + 1] = (temp_vv * scale);
                    visibilities[i + 2] = temp_ww * scale;
                    visibilities[i + 3] = temp_real;
                    visibilities[i + 4] = temp_imag;
                    visibilities[i + 5] = temp_weight;
                }
            }
            
            fclose(file);
        }
        else
            printf("NO VISIBILITY FILE\n");
    }
    else
        visibilities = malloc(sizeof (GLfloat) * config.numVisibilityParams * config.visibilityCount);
    
    GLuint vertexShader;
    
    if(config.useHeavyInterpolation)
        vertexShader = createShader(GL_VERTEX_SHADER, VERTEX_SHADER);
    else
        vertexShader = createShader(GL_VERTEX_SHADER, VERTEX_SHADER_SNAP);
    
    GLuint fragmentShader;
    if(config.fragShaderType == Radial)
        fragmentShader = createShader(GL_FRAGMENT_SHADER, FRAGMENT_SHADER_RADIAL);
    else if(config.fragShaderType == Reflect)
    {
        char *fragmentReflectVec = (config.numVectorElements == 2)
            ? FRAGMENT_SHADER_REFLECT_VEC2 : FRAGMENT_SHADER_REFLECT_VEC4; 
        fragmentShader = createShader(GL_FRAGMENT_SHADER, fragmentReflectVec);
    }
    else
        fragmentShader = createShader(GL_FRAGMENT_SHADER, FRAGMENT_SHADER);
    
    sProgram = createProgram(vertexShader, fragmentShader);
    sLocPosition = glGetAttribLocation(sProgram, "position");
    sComplex = glGetAttribLocation(sProgram, "complex");
    uShaderTextureKernelHandle = glGetUniformLocation(sProgram, "kernelTex");
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
    
    GLint gridInternalFormat = (config.numVectorElements == 2) ? GL_RG32F : GL_RGBA32F;
    GLenum gridVectorElements = (config.numVectorElements == 2) ? GL_RG : GL_RGBA;
    
    glTexImage2D(GL_TEXTURE_2D, 0, gridInternalFormat, config.renderDimension, config.renderDimension,
            0, gridVectorElements, GL_FLOAT, gridBuffer);

    glGenFramebuffers(1, idArray);
    fboID = idArray[0];
    glBindFramebuffer(GL_FRAMEBUFFER, fboID);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, textureID, 0);
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
    glBindTexture(GL_TEXTURE_2D, 0);
    
    KERNEL_DIM = GL_TEXTURE_3D;
    
    if(config.fragShaderType == Radial)
    {
        printf(">>> Using RADIAL based Fragment Shader...\n");
        kernelBuffer = calloc((config.kernelTexSize/2) * config.wProjectNumPlanes, sizeof(FloatComplex));
        KERNEL_DIM = GL_TEXTURE_2D;
    }
    else if(config.fragShaderType == Reflect)
    {
        printf(">>> Using REFLECT based Fragment Shader...\n");
        kernelBuffer = calloc((config.kernelTexSize/2) * (config.kernelTexSize/2) * config.wProjectNumPlanes, sizeof(FloatComplex));
    }
    else
    {
        printf(">>> Using FULL CUBE based Fragment Shader...\n");
        kernelBuffer = calloc(config.kernelTexSize * config.kernelTexSize * config.wProjectNumPlanes, sizeof(FloatComplex));
    }
    
    create_w_projection_kernels(kernelBuffer);

    //kernel TEXTURE
    kernelTextureID = idArray[1];
    glBindTexture(KERNEL_DIM, kernelTextureID);
    
    GLfloat samplingMethod = (config.interpolateTextures == 0) ? GL_NEAREST : GL_LINEAR;
    
    if(config.interpolateTextures == 0)
        printf(">>> Using GL_NEAREST for textures...\n");
    else
        printf(">>> Using GL_LINEAR for textures...\n");
    
    glTexParameterf(KERNEL_DIM, GL_TEXTURE_MIN_FILTER, samplingMethod); // linear = better precision
    glTexParameterf(KERNEL_DIM, GL_TEXTURE_MAG_FILTER, samplingMethod); // linear = better precision
    glTexParameteri(KERNEL_DIM, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(KERNEL_DIM, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
  
    if(config.fragShaderType != Radial)
        glTexParameteri(KERNEL_DIM, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE);
    
    float borderColours[] = {0.0f,0.0f,0.0f,1.0f};
    glTexParameterfv(KERNEL_DIM,GL_TEXTURE_BORDER_COLOR, borderColours);
    
    glEnable(KERNEL_DIM);
    
    if(config.fragShaderType == Radial)
        glTexImage2D(GL_TEXTURE_2D, 0,  GL_RG32F, config.kernelTexSize/2, (int) config.wProjectNumPlanes, 0, GL_RG, GL_FLOAT, kernelBuffer);
    else if(config.fragShaderType == Reflect)
        glTexImage3D(GL_TEXTURE_3D, 0,  GL_RG32F, config.kernelTexSize/2, config.kernelTexSize/2, (int) config.wProjectNumPlanes, 0, GL_RG, GL_FLOAT, kernelBuffer);
    else
        glTexImage3D(GL_TEXTURE_3D, 0,  GL_RG32F, config.kernelTexSize, config.kernelTexSize, (int) config.wProjectNumPlanes, 0, GL_RG, GL_FLOAT, kernelBuffer);

    glBindTexture(KERNEL_DIM, 0);    
    glTexEnvi(GL_POINT_SPRITE, GL_COORD_REPLACE, GL_TRUE);
    glPointParameteri(GL_POINT_SPRITE_COORD_ORIGIN,GL_LOWER_LEFT);
    glEnable(GL_POINT_SPRITE);
    glEnable(GL_VERTEX_PROGRAM_POINT_SIZE);
    
    setShaderUniforms();
    glFinish();
    
    printf("SEEMS LIKE ITS ALL SET UP FINE??? \n");
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
    glUniform1f(uGridCenter, ((float) config.renderDimension) / 2.0);
    float centerOffset =  (1.0f / (float) config.renderDimension);
    printf("Center Offset: %f\n", centerOffset);
    glUniform1f(uGridCenterOffset, centerOffset);
    glUniform1f(uWScale, config.wScale);
    glUniform1f(uWStep, 1.0 / (float) config.wProjectNumPlanes);
    glUniform1f(uUVScale, config.uvScale);
    glUniform1f(uNumPlanes, config.wProjectNumPlanes);
    glUseProgram(0);
    
    glUseProgram(sProgramRender);
    glUniform1f(uGridSizeRender, config.renderDimension);
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
        for (int i = 0; i < config.visibilityCount * config.numVisibilityParams; i += config.numVisibilityParams) {
            // U, V, W, Real, Imaginary, Weight
            visibilities[i] =  -(0.0); // right asc
            visibilities[i + 1] = 0.0;
            visibilities[i + 2] = 0.0;
            visibilities[i + 3] = 1.0;
            visibilities[i + 4] = 0.0;
            visibilities[i + 5] = 1.0f;
        }
    }

    glBindFramebuffer(GL_FRAMEBUFFER, fboID);
    if(!config.accumulateMode)
    {   glClearColor(0.0f, 0.0f, 0.0f, 0.0f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    } 
    
    glEnable(GL_BLEND);
    glViewport(0, 0, config.renderDimension, config.renderDimension);
    

    // start vis bind timing here
    glUseProgram(sProgram);
    glBlendEquationSeparate(GL_FUNC_ADD, GL_FUNC_ADD);
    glBlendFuncSeparate(GL_ONE, GL_ONE, GL_ONE, GL_ONE);
    glBindTexture(KERNEL_DIM, kernelTextureID);
    glUniform1i(uShaderTextureKernelHandle, 0);
    glBindBuffer(GL_ARRAY_BUFFER, visibilityBuffer);
    glBufferData(GL_ARRAY_BUFFER, sizeof (GLfloat) * config.numVisibilityParams * config.visibilityCount, visibilities, GL_STATIC_DRAW);
    glEnableVertexAttribArray(sLocPosition);
    glVertexAttribPointer(sLocPosition, 3, GL_FLOAT, GL_FALSE, config.numVisibilityParams*sizeof(GLfloat), 0);
    glEnableVertexAttribArray(sComplex);
    glVertexAttribPointer(sComplex, 3, GL_FLOAT, GL_FALSE, config.numVisibilityParams*sizeof(GLfloat), (void*) (3*sizeof(GLfloat)));
    
    // Begin OpenGL Timing
    int done = 0;
    GLuint64 timerStart, timerEnd;
    GLuint query[2];
    glGenQueries(2, query);
    glQueryCounter(query[0], GL_TIMESTAMP);
    
    // Execute gridding
    glDrawArrays(GL_POINTS, 0, config.visibilityCount);
    glFinish();
    
    // Finish OpenGL Timing
    glQueryCounter(query[1], GL_TIMESTAMP);
   
    // wait until the query results are available
    while (!done) {
        glGetQueryObjectiv(query[1], 
                           GL_QUERY_RESULT_AVAILABLE, 
                           &done);
    }
    // get the query results
    glGetQueryObjectui64v(query[0], GL_QUERY_RESULT, &timerStart);
    glGetQueryObjectui64v(query[1], GL_QUERY_RESULT, &timerEnd);
    
    
    double msTime = (timerEnd - timerStart) / 1000000.0;
    timer.accumulatedTimeMS += msTime;
    timer.iterations++;
    timer.currentAvgTimeMS = timer.accumulatedTimeMS / timer.iterations;
    timer.sumOfSquareDiffTimeMS += pow(msTime - timer.currentAvgTimeMS, 2.0);
    printf("%d) Time Elapsed: %f ms\n", timer.iterations, msTime);
    
    
    
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
    glClearColor(0.0f, 0.0f, 0.0f, 0.0f);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
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

    iterationCount++;
    
    bool dumped = false;
    if(iterationCount == config.displayDumpTime)
    {   
        //printf("Dumping grid from GPU back to host\n");
        glBindFramebuffer(GL_FRAMEBUFFER, fboID);
        iterationCount = 0;
        
        GLenum gridDataFormat = (config.numVectorElements == 2) ? GL_RG : GL_RGBA;
        
        // Ensure OpenGL has finished
        glFinish();
        glReadPixels(0, 0, config.renderDimension, config.renderDimension,  gridDataFormat,
                GL_FLOAT, gridBuffer);
        glFinish();
        
        // This function can be used if you wish to save the gridder results to file
        // Saves convolutional weights, grid real, and grid imaginary
         if(config.saveGridToFile)
             saveGridToFile(config.renderDimension);
        
        glClearColor(0.0f, 0.0f, 0.0f, 0.0f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        glBindFramebuffer(GL_FRAMEBUFFER, 0);
        
        dumped = true;
        totalDumpsPerformed++;
    }
    
    glFinish();
    glutSwapBuffers();
    
    // Terminate program
    if(totalDumpsPerformed == teminationDumpCount)
    {   
        // saveGriddingStats(timingOutputFile);
        printf("Gridding is finished!!!\n\n");
        exit(0);
    }
}

void timerEvent(int value) {
    glutPostRedisplay();
    glutTimerFunc(config.refreshDelay, timerEvent, 0);
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

DoubleComplex complex_scale(const DoubleComplex z, const double scalar)
{
    return (DoubleComplex) {
        .real      = z.real * scalar,
        .imaginary = z.imaginary * scalar
    };
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

DoubleComplex complexScale(DoubleComplex z, double scalar)
{
    return (DoubleComplex) {.real=z.real*scalar, .imaginary=z.imaginary*scalar};
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

void normalize_kernel(DoubleComplex *kernel, const int texture_size, const double support)
{    
    double sum_of_reals = 0.0;
    int index = 0;

    for(int r = 0; r < texture_size; r++)
    {
        for(int c = 0; c < texture_size; c++)
        {
            index = r * texture_size + c;
            sum_of_reals += kernel[index].real;
        }
    }
    
    const double scaleFactor = pow((double) texture_size / support, 2.0) / sum_of_reals;

    for(int r = 0; r < texture_size; r++)
    {
        for(int c = 0; c < texture_size; c++)
        {
            index = r * texture_size + c;
            kernel[index].real = kernel[index].real * scaleFactor;
            kernel[index].imaginary = kernel[index].imaginary * scaleFactor;
        }
    }
}

//void normalizeKernel(DoubleComplex *kernel, int textureSupport, int wFullSupport)
//{    
//    double realSum = 0.0;
//    int index = 0;
//
//    for(int r = 0; r < textureSupport; r++)
//    {
//        for(int c = 0; c < textureSupport; c++)
//        {
//            index = r * textureSupport + c;
//            realSum += kernel[index].real;
//        }
//    }
//    
//    double scaleFactor = pow((double) textureSupport / (double) wFullSupport, 2.0) / realSum;
//
//    for(int r = 0; r < textureSupport; r++)
//    {
//        for(int c = 0; c < textureSupport; c++)
//        {
//            index = r * textureSupport + c;
//            kernel[index].real = kernel[index].real * scaleFactor;
//            kernel[index].imaginary = kernel[index].imaginary * scaleFactor;
//        }
//    }
//}

void normalizeKernelRadial(DoubleComplex *kernel, int resolution, int support)
{
    int halfResolution = resolution/2;
    int start = (resolution * halfResolution) + halfResolution;
    int end = start + halfResolution;
    int r = 0;
    double realSum = 0.0;
    
//    for(int i = start; i < end; i++)
//    {
//        realSum += kernel[i].real * (2.0*M_PI*(r+0.5));
//        r++;
//    }
    
    for(int i = start; i < end; ++i)
        realSum += kernel[i].real * 2.0 * M_PI * ((i - start));      
        
    
    double scaleFactor = pow((double)resolution/(double)support, 2.0) / realSum;

    // Normalize weights
    for(int i = start; i < end; i++)
    {
        kernel[i].real *= scaleFactor;
        kernel[i].imaginary *= scaleFactor;
    }
}

void create_w_projection_kernels(FloatComplex *w_textures)
{
    const int oversample = 4;
    const int conv_size = config.kernelResolutionSize;
    const int inner = conv_size / oversample;
    const int grid_size = config.gridDimension;
    const int image_size = config.imageSize;
    const int texture_size = config.kernelTexSize;
    const int number_w_planes = config.wProjectNumPlanes;
    
    const double fov = config.fieldOfView;
    const double max_l = sin(0.5 * fov);
    const double sampling = ((2.0 * max_l * oversample) / image_size) * ((double) grid_size / (double) conv_size);
    const double w_scale = config.wScale;
    const double min_support = config.kernelMinFullSupport;
    const double max_support = config.kernelMaxFullSupport;
    const double w_to_max_support_ratio = config.wToMaxSupportRatio;
    
    if((int) max_support * oversample > conv_size) // cannot fit within chosen resolution
    {
        printf("conv size: %d\n", conv_size);
        printf("oversample: %d\n", oversample);
        printf("max_support: %f\n", max_support);
        printf("max_support * os: %d\n", (int) max_support * oversample);
        
        printf(">>> ERROR: Resolution too small to contain product of max support"
                "and oversample, exiting...\n");
        exit(0);
    }
    
    DoubleComplex *screen = calloc(conv_size * conv_size, sizeof(DoubleComplex));
    DoubleComplex *texture = calloc(texture_size * texture_size, sizeof(DoubleComplex));
    double *taper = calloc(inner, sizeof(double)); // 1D Prolate Spheroidal
    
    // Populate 1D Prolate Spheroidal
    for(int taper_i = 0; taper_i < inner; ++taper_i)
    {
        double taper_nu = fabs(calculate_window_stride(taper_i, inner));
        taper[taper_i] = prolate_spheroidal(taper_nu);
    }
    
    const int plane = 0;
    
    printf(">>> UPDATE: Creating w projection kernels...\n");
    for(int iw = 0; iw < 1; iw++)
    {
        const double w = iw * iw / w_scale;
        const double support = calculate_support(w, min_support, w_to_max_support_ratio);
        
        printf("\n>>> UPDATE: Creating kernel number %d (w = %f, support = %f)...\n", 
                iw, w, support);
        
        // Zero out placeholder planes
        memset(screen, 0, conv_size * conv_size * sizeof(DoubleComplex));
        memset(texture, 0, texture_size * texture_size * sizeof(DoubleComplex));
        
        // Generate screen
        printf(">>> UPDATE: Generating phase screen...\n");
        generate_phase_screen(iw, conv_size, inner, sampling, w_scale, taper, screen);
        
        if(iw == plane)
            saveKernelToFile("data/wproj_%f_phase_screen_%d.csv", w, conv_size, screen);
        
        printf(">>> UPDATE: Executing Fourier Transform...\n");
        fft_shift_in_place(screen, conv_size);
        fft_2d(screen, conv_size);
        fft_shift_in_place(screen, conv_size);
        
        if(iw == plane)
            saveKernelToFile("data/wproj_%f_after_fft_%d.csv", w, conv_size, screen);
        
        printf(">>> UPDATE: Performing bicubic interpolation...\n");
        interpolate_kernel(screen, texture, conv_size, texture_size, support, oversample);
        
        if(iw == plane)
            saveKernelToFile("data/wproj_%f_after_interpolated_%d.csv", w, texture_size, texture);

        // Normalize the kernel
        printf(">>> UPDATE: Performing kernel normalization...\n");
        normalize_kernel(texture, texture_size, support);
        
        if(iw == plane)
            saveKernelToFile("data/wproj_%f_normalized_%d_from_%d.csv", w, texture_size, texture); 
        
        // Bind scaled and normalized texture to 3D texture block
        printf(">>> UPDATE: Clipping produced kernel, storing into 3D texture block...\n");
        const int texture_size_half = texture_size / 2;
        for(int y = texture_size_half; y < texture_size; y++)
        {
            for(int x = texture_size_half; x < texture_size; x++)
            {
                DoubleComplex sample = texture[y * texture_size + x];
                int index = (iw * texture_size_half * texture_size_half)
                    + ((y - texture_size_half) * texture_size_half)
                    + (x - texture_size_half);
                
                w_textures[index] = (FloatComplex) {
                    .real      = (float) sample.real, 
                    .imaginary = (float) sample.imaginary
                };
            }
        }  
    }
    
    free(taper);
    free(texture);
    free(screen);
}

void generate_phase_screen(const int iw, const int conv_size, const int inner,
    const double sampling, const double w_scale, double *taper, DoubleComplex *screen)
{
    double f = (2.0 * M_PI * iw * iw) / w_scale;
    int conv_size_half = conv_size / 2;
    int inner_half = inner / 2;
    
    for(int iy = 0 ; iy < inner; ++iy)
    {
        double taper_y = taper[iy];
        double m = sampling * (double) iy;
        double msq = m*m;
        // int offset = (iy > -1 ? iy : (iy + conv_size)) * conv_size;
        
        for(int ix = 0; ix < inner; ++ix)
        {
            double l = sampling * (double) ix;
            double rsq = l * l + msq;
            if (rsq < 1.0) {
                double taper_x = taper[ix];
                double taper = taper_x * taper_y;
                int index = (conv_size_half - inner_half + iy) * conv_size + (conv_size_half - inner_half + ix);
                double phase = f * (sqrt(1.0 - rsq) - 1.0);
                screen[index] = (DoubleComplex) {
                    .real      = taper * cos(phase),
                    .imaginary = taper * sin(phase)
                };
            }
        }
    }
}

double calculate_support(const double w, const int min_support, const double w_max_support_ratio)
{   
    return fabs(w_max_support_ratio * w) + min_support;
}

void fft_2d(DoubleComplex *matrix, int number_channels)
{
    // Calculate bit reversed indices
    int* bit_reverse_indices = calloc(number_channels, sizeof(int));
    calc_bit_reverse_indices(number_channels, bit_reverse_indices);
    DoubleComplex *reverse_buffer = calloc(number_channels * number_channels, sizeof(DoubleComplex));
    
    for(int row = 0; row < number_channels; ++row)
        for(int col = 0; col < number_channels; ++col)
        {
            int row_reverse = bit_reverse_indices[row];
            int col_reverse = bit_reverse_indices[col];
            int bit_reverse_index = row_reverse * number_channels + col_reverse;
            int matrix_index = row * number_channels + col;
            reverse_buffer[matrix_index] = matrix[bit_reverse_index];
            // printf("%d -> %d\n", matrix_index, bit_reverse_index);
        }
    
    memcpy(matrix, reverse_buffer, number_channels * number_channels * sizeof(DoubleComplex));
    free(reverse_buffer);
    free(bit_reverse_indices);
    
    for(int m = 2; m <= number_channels; m *= 2)
    {
        DoubleComplex omegaM = (DoubleComplex) {
            .real      = cos(M_PI * 2.0 / m),
            .imaginary = sin(M_PI * 2.0 / m)
        };
        
        for(int k = 0; k < number_channels; k += m)
        {
            for(int l = 0; l < number_channels; l += m)
            {
                DoubleComplex x = (DoubleComplex) {.real = 1.0, .imaginary = 0.0};
                
                for(int i = 0; i < m / 2; i++)
                {
                    DoubleComplex y = (DoubleComplex) {.real = 1.0, .imaginary = 0.0};
                    
                    for(int j = 0; j < m / 2; j++)
                    {   
                        // Perform 2D butterfly operation in-place at (k+j, l+j)
                        int in00Index = (k+i) * number_channels + (l+j);
                        DoubleComplex in00 = matrix[in00Index];
                        int in01Index = (k+i) * number_channels + (l+j+m/2);
                        DoubleComplex in01 = complexMultiply(matrix[in01Index], y);
                        int in10Index = (k+i+m/2) * number_channels + (l+j);
                        DoubleComplex in10 = complexMultiply(matrix[in10Index], x);
                        int in11Index = (k+i+m/2) * number_channels + (l+j+m/2);
                        DoubleComplex in11 = complexMultiply(complexMultiply(matrix[in11Index], x), y);
                        
                        DoubleComplex temp00 = complexAdd(in00, in01);
                        DoubleComplex temp01 = complexSubtract(in00, in01);
                        DoubleComplex temp10 = complexAdd(in10, in11);
                        DoubleComplex temp11 = complexSubtract(in10, in11);
                        
                        matrix[in00Index] = complexAdd(temp00, temp10);
                        matrix[in01Index] = complexAdd(temp01, temp11);
                        matrix[in10Index] = complexSubtract(temp00, temp10);
                        matrix[in11Index] = complexSubtract(temp01, temp11);
                        y = complexMultiply(y, omegaM);
                    }
                    x = complexMultiply(x, omegaM);
                }
            }
        }
    }
    
    for(int row = 0; row < number_channels; ++row)
        for(int col = 0; col < number_channels; ++col)
        {   
            int matrix_index = row * number_channels + col;
            double reciprocal = 1.0 / (number_channels * number_channels);
            matrix[matrix_index] = complex_scale(matrix[matrix_index], reciprocal);
        }
}

void calc_bit_reverse_indices(int n, int* indices)
{   
    for(int i = 0; i < n; ++i)
    {
        // Calculate index r to which i will be moved
        unsigned int i_prime = i;
        int r = 0;
        for(int j = 1; j < n; j *= 2)
        {
            int b = i_prime & 1;
            r = (r << 1) + b;
            i_prime = (i_prime >> 1);
        }
        indices[i] = r;
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
//void createWProjectionPlanes(FloatComplex *wTextures)
//{                
//    int convolutionSize = config.kernelResolutionSize;
//    int textureSupport = config.kernelTexSize;
//    int numWPlanes = config.wProjectNumPlanes;
//    double wScale = config.wScale;
//    double fov = config.fieldOfView;
//    
//    // Flat two dimension array
//    DoubleComplex *screen = calloc(convolutionSize * convolutionSize, sizeof(DoubleComplex));
//    // Create w screen 
//    DoubleComplex *shift = calloc(convolutionSize * convolutionSize, sizeof(DoubleComplex));
//    
//     // Test variable to output W plane creation steps
//    // (phase screen, after fft, interpolated, normalized)
//    int plane = 0;
//    printf("Num W Planes: %d and wScale %lf\n", numWPlanes, wScale);
//    for(int iw = 0; iw < numWPlanes; iw++)
//    {        
//        // Calculate w term and w specific support size
//        double w = iw * iw / wScale;
//        double fresnel = w * ((0.5 * fov)*(0.5 * fov));
//        int wFullSupport = calcWFullSupport(w, config.wToMaxSupportRatio, config.kernelMinFullSupport);
//        printf("Creating W Plane: (%d) For w = %f, field of view = %f, " \
//                "fresnel number = %f, full w support: %d texSupport : %d\n", iw, w, fov, fresnel, wFullSupport, textureSupport);
//        
//        // Create Phase Screen
//        createPhaseScreen(convolutionSize, screen, w, fov, wFullSupport);
//        
//        if(iw == plane)
//            saveKernelToFile("data/wproj_%f_phase_screen_%d.csv", w, convolutionSize, screen);
//        
//        // Perform shift and inverse FFT of Phase Screen
//        fft2dShift(convolutionSize, screen, shift);
//        inverseFFT2dVectorRadixTransform(convolutionSize, shift, screen);
//        fft2dShift(convolutionSize, screen, shift);
//        
//        
//        if(iw == plane)
//            saveKernelToFile("data/wproj_%f_after_fft_%d.csv", w, convolutionSize, shift);
//            
//        // Interpolate w projection kernel down to texture support dimensions
//         DoubleComplex *interpolated = calloc(textureSupport * textureSupport, sizeof(DoubleComplex));
//         interpolateKernel(shift, interpolated, convolutionSize, textureSupport);
//         
//         if(iw == plane)
//             saveKernelToFile("data/wproj_%f_after_interpolated_%d.csv", w, textureSupport, interpolated);
//
//        // Normalize the kernel
//        normalizeKernel(interpolated, textureSupport, wFullSupport);
//        
//        if(iw == plane)
//            saveKernelToFile("data/wproj_%f_normalized_%d_from_%d.csv", w, textureSupport, interpolated); 
//        
//        
//        // Bind interpolated kernel to texture matrix
//        int startIndex = (config.fragShaderType == Reflect) ? textureSupport/2 : 0;
//        int kernelWidth = (config.fragShaderType == Reflect) ? textureSupport/2 : textureSupport;
//            
//        for(int y = startIndex; y < textureSupport; y++)
//        {
//            for(int x = startIndex; x < textureSupport; x++)
//            {
//                DoubleComplex interpWeight = interpolated[y * textureSupport + x];
//                FloatComplex weight = (FloatComplex) {.real = (float) interpWeight.real, .imaginary = (float) interpWeight.imaginary};
//                int index = (iw * kernelWidth * kernelWidth) + ((y-startIndex) * kernelWidth) + (x-startIndex);
//                wTextures[index] = weight;
//            }
//
//        }  
//         
//        free(interpolated);
//        memset(screen, 0, convolutionSize * convolutionSize * sizeof(DoubleComplex));
//        memset(shift, 0, convolutionSize * convolutionSize * sizeof(DoubleComplex));
//    }
//    
//    free(screen);
//    free(shift);
//}

//void createPhaseScreenNew(int iw, int full_support, int conv_size, 
//    double sampling, double w_scale, DoubleComplex *screen)
//{        
//    double f = (2.0 * M_PI * iw * iw) / w_scale;
//    int conv_size_half = conv_size / 2;
//    int half_support = (full_support - 1) / 2;
//    
//    for(int iy = 0; iy < full_support; ++iy)
//    {
//        double stride_y = fabs((iy - full_support / 2) / ((double) full_support / 2.0));
//        double taper_y = calcSpheroidalWeight(stride_y);
//        printf(">>> %f\n", stride_y);
//        double m = sampling * (double) (iy - half_support);
//        double msq = m*m;
//        for(int ix = 0; ix < full_support; ++ix)
//        {
//            double l = sampling * (double) (ix - half_support);
//            double rsq = l * l + msq;
//            if (rsq < 1.0) {
//                double stride_x = fabs((ix - full_support / 2) / ((double) full_support / 2.0));
//                double taper_x = calcSpheroidalWeight(stride_x);
//                double taper = taper_x * taper_y;               
//                int row_index = conv_size_half + iy - half_support;
//                int col_index = conv_size_half + ix - half_support;
//                int index = row_index * conv_size + col_index;
//                double phase = f * (sqrt(1.0 - rsq) - 1.0);
//                screen[index] = (DoubleComplex) {
//                    .real      = taper * cos(phase),
//                    .imaginary = taper * sin(phase)
//                };
//            }
//        }
//    }
//}

//void createPhaseScreen(int resolutionFullSupport, DoubleComplex *screen, double w, double fieldOfView, int wFullSupport)
//{        
//    int resolutionHalfSupport = resolutionFullSupport/2;
//    int paddedWFullSupport = wFullSupport;
//    int paddedWHalfSupport = paddedWFullSupport/2;
//    int index = 0;
//    double taper, taperY, nuX, nuY, radius;
//    double l, m, lsq, rsq, phase;
//    
//    for(int iy = 0; iy < paddedWFullSupport; iy++)
//    {
//        l = (((double) iy-(paddedWHalfSupport)) / (double) wFullSupport) * fieldOfView;
//        lsq = l*l;
//        phase = 0.0;
//        
//        nuY = fabs(calcAndrewShift(iy, paddedWFullSupport)); 
//        if(config.fragShaderType != Radial)
//            taperY = calcSpheroidalWeight(nuY);
//        
//        for(int ix = 0; ix < paddedWFullSupport; ix++)
//        {
//            m = (((double) ix-(paddedWHalfSupport)) / (double) wFullSupport) * fieldOfView;
//            rsq = lsq+(m*m);
//            index = (iy+(resolutionHalfSupport-paddedWHalfSupport)) * resolutionFullSupport
//                    + (ix+(resolutionHalfSupport-paddedWHalfSupport));
//
//            nuX = fabs(calcAndrewShift(ix, paddedWFullSupport));
//            taper = taperY * calcSpheroidalWeight(nuX);
//            
//            if(rsq < 1.0)
//            {
//                phase = w * (1.0 - sqrt(1.0 - rsq));
//                screen[index] = complexConjugateExp(phase);
//
//            }
//            
//            if(rsq == 0.0)
//                screen[index] = (DoubleComplex) {.real = 1.0, .imaginary = 0.0};
//                
//            screen[index].real *= taper;
//            screen[index].imaginary *= taper;
//        }
//    }
//}

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
 * Function: prolate_spheroidal 
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
double prolate_spheroidal(double nu)
{   
    static double p[] = {0.08203343, -0.3644705, 0.627866, -0.5335581, 0.2312756,
        0.004028559, -0.03697768, 0.1021332, -0.1201436, 0.06412774};
    static double q[] = {1.0, 0.8212018, 0.2078043,
        1.0, 0.9599102, 0.2918724};
    
    int part, sp, sq;
    double nuend, delta, top, bottom;
    
    if(nu >= 0.0 && nu < 0.75)
    {
        part = 0;   
        nuend = 0.75;
    }
    else if(nu >= 0.75 && nu < 1.0)
    {
        part = 1;
        nuend = 1.0;
    }
    else
        return 0.0;

    delta = nu * nu - nuend * nuend;
    sp = part * 5;
    sq = part * 3;
    top = p[sp];
    bottom = q[sq];
    
    for(int i = 1; i < 5; i++)
        top += p[sp+i] * pow(delta, i);
    for(int i = 1; i < 3; i++)
        bottom += q[sq+i] * pow(delta, i);
    return (bottom == 0.0) ? 0.0 : top/bottom;
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

void fft_shift_in_place(DoubleComplex *matrix, const int size)
{
    for(int row = 0; row < size; ++row)
        for(int col = 0; col < size; ++col)
        {
            const int matrix_index = row * size + col;
            const double scalar = 1 - 2 * ((row + col) & 1);
            matrix[matrix_index] = complex_scale(matrix[matrix_index], scalar);
        }
}

double calculate_window_stride(const int index, const int width)
{
    return (index - width / 2) / ((double) width / 2.0);
}

double calcAndrewShift(int index, int fullSupport)
{
    return (index - fullSupport / 2) / ((double) fullSupport / 2.0);
}

double calcInterpolateShift(double index, double width)
{
    return -1.0 + ((2.0 * index + 1.0) / width);
}

int calcRelativeIndex(double x, double width)
{
    int offset = 0;//(x < 0.0) ? 1 : 2;
    return ((int) round(((x+1.0)/2.0) * (width-offset))) ;
}

double calcSphrShift(double index, double width)
{   
    return -1.0 + index * (2.0/width);
}

void saveKernelToFile(char* filename, double w, int support, DoubleComplex* data)
{
    char *buffer[100];
    sprintf(buffer, filename, w, support, config.kernelResolutionSize);
    FILE *file = fopen(buffer, "w");
    for(int r = 0; r < support; r++)
    {
        for(int c = 0; c < support; c++)
        {
            //if(r == support/2)
                fprintf(file, "%.15f ", data[r * support + c].real);
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
    printf(">>> Saving grid to file...\n");
    // Save to file (real portion)
    FILE *file_real = fopen(config.outputGridReal, "w");
    FILE *file_imag = fopen(config.outputGridImag, "w");
    
    int gridRealIndex = (config.numVectorElements == 2) ? 0 : 1;
    int gridImagIndex = (config.numVectorElements == 2) ? 1 : 2;
    
    double realSum = 0.0, imagSum = 0.0;
    int index = 0;
    //FILE *file_weight = fopen("output/grid_weight.csv", "w");
    for(int r = 0; r < config.renderDimension; r++)
    {
        for(int c = 0; c < config.renderDimension * config.numVectorElements; c += config.numVectorElements)
        {   
            index = (r * ((int) config.renderDimension) * config.numVectorElements) + c;
            //fprintf(file_weight, "%+f, ", gridBuffer[(r*(int)config.renderDimension*4)+c]);
            if(c < (config.renderDimension * config.numVectorElements - config.numVectorElements))
            {   fprintf(file_real, "%.15f,", gridBuffer[index+gridRealIndex]);
                fprintf(file_imag, "%.15f,", gridBuffer[index+gridImagIndex]);
            }
            else
            {   fprintf(file_real, "%.15f", gridBuffer[index+gridRealIndex]);
                fprintf(file_imag, "%.15f", gridBuffer[index+gridImagIndex]); 
            }
            realSum += gridBuffer[index+gridRealIndex];
            imagSum += gridBuffer[index+gridImagIndex];
        }
        fprintf(file_real, "\n");
        fprintf(file_imag, "\n");
        //fprintf(file_weight, "\n");
    }
    fclose(file_real);
    fclose(file_imag);
    //fclose(file_weight);
    
    printf("RealSum: %.10f, ImagSum: %f\n", realSum, imagSum);
    printf("Grid has been output to file\n");
}

void interpolate_kernel(DoubleComplex *screen, DoubleComplex *texture, 
    const int screen_size, const int texture_size, const double support,
        const int oversample)
{
    const int oversampled_support = oversample * support;
    
    // Determine "distance" between source samples in range [-1.0, 1.0]
    double screen_shift = 2.0 / (oversampled_support - 1.0);
    
    // Storage for neighbours, synthesized points
    DoubleComplex n[16], p[4];
    
    // Neighbours shift values (rs = row shift, cs = col shift)
    double *rs = calloc(16, sizeof(double));
    double *cs = calloc(16, sizeof(double));
    double row_shift, col_shift = 0.0;
    
    for(int r = 0; r < texture_size; r++)
    {        
        // Determine relative shift for interpolation row [-1.0, 1.0]
        row_shift = calcInterpolateShift((double)r, (double) texture_size);
        
        for(int c = 0; c < texture_size; c++)
        {
            memset(rs, 0, 16 * sizeof(double));
            memset(cs, 0, 16 * sizeof(double));
            
            // Determine relative shift for interpolation col [-1.0, 1.0]
            col_shift = calcInterpolateShift((double)c, (double) texture_size);
            
            // gather 16 neighbours
            getBicubicNeighbours(row_shift, col_shift, n, rs, cs, screen_size,
                screen, oversampled_support);
            
            // interpolate intermediate samples
            p[0] = interpolateSample(n[0], n[1], n[2], n[3],
                cs[0], cs[1], cs[2], cs[3], screen_shift, col_shift);
            p[1] = interpolateSample(n[4], n[5], n[6], n[7],
                cs[4], cs[5], cs[6], cs[7], screen_shift, col_shift);
            p[2] = interpolateSample(n[8], n[9], n[10], n[11],
                cs[8], cs[9], cs[10], cs[11], screen_shift, col_shift);
            p[3] = interpolateSample(n[12], n[13], n[14], n[15],
                cs[12], cs[13], cs[14], cs[15], screen_shift, col_shift);
            
            // interpolate final sample
            texture[r * texture_size + c] = interpolateSample(p[0], p[1], p[2], p[3],
               rs[1], rs[5], rs[9], rs[13], screen_shift, row_shift);
        }
    }
    
    free(rs);
    free(cs);
}

//void interpolateKernel(DoubleComplex *source, DoubleComplex *destination, 
//    int sourceSupport, int destinationSupport)
//{
//    // Determine "distance" between source samples in range [-1.0, 1.0]
//    double sourceShift = 2.0/(sourceSupport-1.0);
//    // Storage for neighbours, synthesized points
//    DoubleComplex n[16], p[4];
//    // Neighbours shift values (rs = row shift, cs = col shift)
//    double rs[16], cs[16];
//    double rowShift, colShift = 0.0;
//    
//    for(int r = 0; r < destinationSupport; r++)
//    {
//        // Determine relative shift for interpolation row [-1.0, 1.0]
//        rowShift = calcInterpolateShift((double)r, (double)destinationSupport);
//        
//        for(int c = 0; c < destinationSupport; c++)
//        {
//            // Determine relative shift for interpolation col [-1.0, 1.0]
//            colShift = calcInterpolateShift((double)c, (double)destinationSupport);
//            
//            // gather 16 neighbours
//            getBicubicNeighbours(rowShift, colShift, n, rs, cs, sourceSupport, source);
//            // interpolate intermediate samples
//            p[0] = interpolateSample(n[0], n[1], n[2], n[3],
//                cs[0], cs[1], cs[2], cs[3], sourceShift, colShift);
//            p[1] = interpolateSample(n[4], n[5], n[6], n[7],
//                cs[4], cs[5], cs[6], cs[7], sourceShift, colShift);
//            p[2] = interpolateSample(n[8], n[9], n[10], n[11],
//                cs[8], cs[9], cs[10], cs[11], sourceShift, colShift);
//            p[3] = interpolateSample(n[12], n[13], n[14], n[15],
//                cs[12], cs[13], cs[14], cs[15], sourceShift, colShift);
//            
//            // interpolate final sample
//            destination[r * destinationSupport + c] = interpolateSample(p[0], p[1], p[2], p[3],
//               rs[1], rs[5], rs[9], rs[13], sourceShift, rowShift);
//        }
//    }
//}

void getBicubicNeighbours(double rowShift, double colShift, DoubleComplex *n, double *rs, double *cs,
        int screen_size, DoubleComplex *screen, const int oversampled_support)
{
    const int offset = (screen_size / 2 - oversampled_support / 2);
    // determine where to start locating neighbours in source matrix
    int x = calcRelativeIndex(colShift, (double) oversampled_support);
    int y = calcRelativeIndex(rowShift, (double) oversampled_support);
    
    // printf("x = %f, y = %f\n", colShift, rowShift);
   // printf("x = %d, y = %d\n", x, y);
    
    // counter for active neighbour
    int nIndex = 0;
    // define neighbour boundaries
    int rStart = (rowShift < 0.0) ? y-1 : y-1;
    int rEnd = (rowShift < 0.0) ? y+3 : y+3;
    int cStart = (colShift < 0.0) ? x-1 : x-1;
    int cEnd = (colShift < 0.0) ? x+3 : x+3;
    
    // gather 16 neighbours
    for(int r = rStart; r < rEnd; r++)
    {   
        for(int c = cStart; c < cEnd; c++)
        {
            // set row and col shifts for neighbour
//            rs[nIndex] = (rowShift < 0.0) ? calcSphrShift(r, oversampled_support+1) : calcSphrShift(r, oversampled_support+1);
//            cs[nIndex] = (colShift < 0.0) ? calcSphrShift(c, oversampled_support+1) : calcSphrShift(c, oversampled_support+1);       
            
            rs[nIndex] = (true) ? calcSphrShift(r, oversampled_support) : calcSphrShift(r, oversampled_support);
            cs[nIndex] = (true) ? calcSphrShift(c, oversampled_support) : calcSphrShift(c, oversampled_support);
            
            // neighbour falls out of bounds
            if(r < 0 || c < 0 || r >= oversampled_support || c >= oversampled_support)
                n[nIndex] = (DoubleComplex) {.real = 0.0, .imaginary = 0.0};
            // neighbour exists
            else
                n[nIndex] = screen[(r+offset) * screen_size + (c+offset)];
            
//             printf("[r: %f, c: %f] ", rs[nIndex], cs[nIndex]);
//             printf("[r: %d, c: %d] ", r, c);
            
            nIndex++;
        }
//         printf("\n");
    }   
    // printf("\n\n");
}

//void getBicubicNeighbours(double rowShift, double colShift, DoubleComplex *n, double *rs, double *cs,
//        int screen_size, DoubleComplex *screen, const int oversampled_support)
//{
//    const int offset = screen_size / 2 - oversampled_support / 2;
//    // determine where to start locating neighbours in source matrix
//    int x = calcRelativeIndex(colShift, (double) oversampled_support);
//    int y = calcRelativeIndex(rowShift, (double) oversampled_support);
//    
//    // printf("x = %d, y = %d\n", x,y);
//    // counter for active neighbour
//    int nIndex = 0;
//    // define neighbour boundaries
//    int rStart = (rowShift < 0.0) ? y-1 : y-2;
//    int rEnd = (rowShift < 0.0) ? y+3 : y+2;
//    int cStart = (colShift < 0.0) ? x-1 : x-2;
//    int cEnd = (colShift < 0.0) ? x+3 : x+2;
//    
//    // gather 16 neighbours
//    for(int r = rStart; r < rEnd; r++)
//    {   
//        for(int c = cStart; c < cEnd; c++)
//        {
//            // set row and col shifts for neighbour
//            rs[nIndex] = (rowShift < 0.0) ? calcSphrShift(r-1, oversampled_support-1) : calcSphrShift(r, oversampled_support-1);
//            cs[nIndex] = (colShift < 0.0) ? calcSphrShift(c-1, oversampled_support-1) : calcSphrShift(c, oversampled_support-1);            
//            // neighbour falls out of bounds
//            if(r < 0 || c < 0 || r >= oversampled_support || c >= oversampled_support)
//                n[nIndex] = (DoubleComplex) {.real = 0.0, .imaginary = 0.0};
//            // neighbour exists
//            else
//                n[nIndex] = screen[(r+offset) * screen_size + (c+offset)];
//
//            nIndex++;
//        }
//    }   
//}

DoubleComplex interpolateSample(DoubleComplex z0, DoubleComplex z1, 
    DoubleComplex z2, DoubleComplex z3, double x0, double x1, double x2,
    double x3, double h, double x)
{
    double hCube = pow(h, 3.0);
    double scale0 = -(x-x1)*(x-x2)*(x-x3)/(6.0*hCube);
    double scale1 = (x-x0)*(x-x2)*(x-x3)/(2.0*hCube);
    double scale2 = -(x-x0)*(x-x1)*(x-x3)/(2.0*hCube);
    double scale3 = (x-x0)*(x-x1)*(x-x2)/(6.0*hCube);
   
    DoubleComplex z = complexScale(z0, scale0);
    z = complexAdd(z, complexScale(z1, scale1));
    z = complexAdd(z, complexScale(z2, scale2));
    z = complexAdd(z, complexScale(z3, scale3));
    
    return z;
}

void saveGriddingStats(char *filename)
{
    FILE *f = fopen(filename, "w");
    
    if(f != NULL)
    {
        fprintf(f, ">>> Gridding Statistics <<<\n");
        fprintf(f, "Grid Dimension: %d\n", config.gridDimension);
        fprintf(f, "Render Dimension: %d\n", config.renderDimension);
        fprintf(f, "Texture Dimension: %d\n", config.kernelTexSize);
        fprintf(f, "Resolution Dimension: %d\n", config.kernelResolutionSize);
        fprintf(f, "Cell Size Rads: %.15f\n", config.cellSizeRad);
        fprintf(f, "Max W: %.15f\n", config.wProjectionMaxW);
        fprintf(f, "W Scale: %.15f\n", config.wScale);
        fprintf(f, "Kernel Min Support: %f\n", config.kernelMinFullSupport);
        fprintf(f, "Kernel Max Support: %f\n", config.kernelMaxFullSupport);
        fprintf(f, "Texture interpolation: %d\n", config.interpolateTextures);
        fprintf(f, "Average Gridding Time: %f\n", ((double)timer.accumulatedTimeMS/timer.iterations));
        fprintf(f, "STDev Gridding Time: %f\n", sqrt(timer.sumOfSquareDiffTimeMS/(timer.iterations-1)));
        fclose(f);
        printf(">>> SUcCESS: Gridding stats saved to file\n");
    }
    else
        printf(">>> ERR: Unable to save gridding stats to file\n");
}