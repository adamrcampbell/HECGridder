
/* 
 * File:   gridder.c
 * Author: adam
 *
 * Created on 1 August 2017, 11:23 AM
 */

//31 million visibilities = 620mb
//3k squared grid         = 32mb

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
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_vector.h>
#include <gsl/gsl_linalg.h>
#include <gsl/gsl_eigen.h>
#include <gsl/gsl_sf.h>     

#include "gridder.h"
#include "gpu.h"


/*
 TODO LIST for this week!!
 * - Make some random depths, ie w depth matches the test data
 * - W projection properly,
 * - kernel size properly, work out how W affects kernel size
 * - get it so that its the HEC gridder converting UVW to grid cords
 * - config list... Easy to config the gridder for others
 * - refactor and tidy up code base!!
 */

/* ASK THE ENZ LIST!
 
 * HOW CAN WE CENTER THIS ON A CENTER PIXEL
 * 
 * SHOULD THE GRID BE LINEAR OR NEAREST?? KERNEL OBV LINEAR BUT THE GRID TEX??
 
 */

/*--------------------------------------------------------------------
 *   GUI CONFIG
 *-------------------------------------------------------------------*/
static GLfloat guiRenderBounds[8];


/*--------------------------------------------------------------------
 *   PROLATE SPHEROIDAL CONFIG
 *-------------------------------------------------------------------*/
static struct SpheroidalFunction spheroidal;

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
static GLuint uGridSize;
static GLuint uGridSizeRender;
static GLuint uWScale;
static GLuint uUVScale; 

static GLuint fboID;
static GLuint textureID;
static GLuint kernalTextureID;

static GLfloat* gridBuffer;
static FloatComplex* kernelBuffer;
static GLuint* visibilityIndices;
static GLfloat* visibilities;

int iterationCount = 0;

int counter =0;
int counterAverage;
double sumTimeReal;
float sumTimeProcess;
int val;
struct timeval timeCallsReal;
int timeCallsProcess;
bool toggle = 0;  
int windowDisplay;

// W projection and gridding configuration
Config config;


void initConfig(void) {
    // Global
    windowDisplay = 900;
    config.kernelTexSize = 128;
    config.gridDimension = 18000.0f;
    config.kernelMaxFullSupport = (44.0f * 2.0f) + 1.0f;
    config.kernelMinFullSupport = (4.0f * 2.0f) + 1.0f;
    config.visibilityCount = 1;
    // config.visibilityCount = 1;//31395840;
    config.numVisibilityParams = 5;
    config.visibilitiesFromFile = true;
    config.displayDumpTime = 10;
    config.visibilitySourceFile = "el82-70.txt"; //"el82-70_vis.txt";
    // Gui
    config.refreshDelay = 0;
    GLfloat renderTemp[8] = {
        -config.gridDimension/2.0f, -config.gridDimension/2.0f,
        -config.gridDimension/2.0f, config.gridDimension/2.0f,
        config.gridDimension/2.0f, -config.gridDimension/2.0f,
        config.gridDimension/2.0f, config.gridDimension/2.0f
    };
    memcpy(guiRenderBounds, renderTemp, sizeof (guiRenderBounds));
    
    // Prolate
    config.prolateC = 3 * M_PI;
    config.prolateAlpha = 1;
    config.prolateNumTerms = 16; // default: 16
    
    // W-Projection
    config.wProjectionMaxW = 7000.0f;
    config.wProjectionMaxPlane = 339.0f; //714
    config.wScale = (config.wProjectionMaxPlane * config.wProjectionMaxPlane) / config.wProjectionMaxW;
    config.wProjectionStep = config.wProjectionMaxW / config.wScale;
//    config.wProjectNumPlanes = (unsigned int) ceilf((fabsf(config.wProjectionMaxW)*2+config.wProjectionStep)/config.wProjectionStep);
    config.cellSize = 0.000006;
    config.fieldOfView = config.cellSize * config.gridDimension; 
}

void initGridder(void) {
    
    kernelBuffer = malloc(sizeof (FloatComplex) * config.kernelTexSize * config.kernelTexSize * (config.wProjectionMaxPlane+1.0f));
//    createWPlanes();
//    exit(0);
    
    initSpheroidal();
    
    glClampColorARB(GL_CLAMP_VERTEX_COLOR_ARB, GL_FALSE);
    glClampColorARB(GL_CLAMP_READ_COLOR_ARB, GL_FALSE);
    glClampColorARB(GL_CLAMP_FRAGMENT_COLOR_ARB, GL_FALSE);

    srand(time(NULL));
    int size = 4 * config.gridDimension*config.gridDimension;
    gridBuffer = (GLfloat*) malloc(sizeof (GLfloat) * size);

    for (int i = 0; i < size; i += 4) {
        gridBuffer[i] = 0.0f;
        gridBuffer[i + 1] = 0.0f;
        gridBuffer[i + 2] = 0.0f;
        gridBuffer[i + 3] = 0.0f;
    }

    if(config.visibilitiesFromFile)
    {
        FILE *file = fopen(config.visibilitySourceFile, "r");
        
        if(file != NULL)
        {
            int visCount = 0;
            fscanf(file, "%d\n", &visCount);
            config.visibilityCount = visCount;
            printf("READING %d number of visibilities from file ",config.visibilityCount);
            visibilities = malloc(sizeof (GLfloat) * config.numVisibilityParams * config.visibilityCount);
            float temp_uu, temp_vv, temp_ww = 0.0f;
            float temp_real, temp_imag = 0.0f;
            for(int i = 0; i < config.visibilityCount * config.numVisibilityParams; i+=config.numVisibilityParams)
            {
                fscanf(file, "%f %f %f %f %f\n", &temp_uu, &temp_vv, &temp_ww, &temp_real, &temp_imag);
                
//                int randomKernel = (int)((float)rand()/RAND_MAX * 44)+44;
//
//                while(randomKernel % 2 == 0)
//                {
//                    randomKernel = (int)((float)rand()/RAND_MAX * 44)+44;
//                }
                
                visibilities[i] = temp_uu;
                visibilities[i + 1] = temp_vv;
                visibilities[i + 2] = temp_ww;//(float) randomKernel;//37.0f; //temp_ww;
                visibilities[i + 3] = temp_real; // remove abs
                visibilities[i + 4] = temp_imag; // remove abs
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
    
   // char buffer[2000];
    //sprintf(buffer, VERTEX_SHADER, config.gridDimension);
    GLuint vertexShader = createShader(GL_VERTEX_SHADER, VERTEX_SHADER);
    GLuint fragmentShader = createShader(GL_FRAGMENT_SHADER, FRAGMENT_SHADER);
    sProgram = createProgram(vertexShader, fragmentShader);
    sLocPosition = glGetAttribLocation(sProgram, "position");
    sComplex = glGetAttribLocation(sProgram, "complex");
    uShaderTextureKernalHandle = glGetUniformLocation(sProgram, "kernalTex");
    uMinSupportOffset = glGetUniformLocation(sProgram, "minSupportOffset");
    uWToMaxSupportRatio = glGetUniformLocation(sProgram, "wToMaxSupportRatio");
    uGridSize = glGetUniformLocation(sProgram, "gridSize");
    uWScale = glGetUniformLocation(sProgram, "wScale");
    uUVScale = glGetUniformLocation(sProgram, "uvScale");

   // sprintf(buffer, VERTEX_SHADER_RENDER, config.gridDimension);
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
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
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
    
    for(int i = 0; i < config.wProjectionMaxPlane; i++)
        createKernel(i);
    
    //kernal TEXTURE
    kernalTextureID = idArray[1];
    glBindTexture(GL_TEXTURE_3D, kernalTextureID);
    glTexParameterf(GL_TEXTURE_3D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameterf(GL_TEXTURE_3D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE);
    glEnable(GL_TEXTURE_3D);
    // width, height, depth
    glTexImage3D(GL_TEXTURE_3D, 0,  GL_RG32F, config.kernelTexSize, config.kernelTexSize, (int) config.wProjectionMaxPlane, 0, GL_RG, GL_FLOAT, kernelBuffer);
    glBindTexture(GL_TEXTURE_3D, 0);
    
    glTexEnvi(GL_POINT_SPRITE, GL_COORD_REPLACE, GL_TRUE);
    glEnable(GL_POINT_SPRITE);
    glEnable(GL_VERTEX_PROGRAM_POINT_SIZE);
    
    
    setShaderUniforms();
    
    counter = 0;
    sumTimeProcess = 0.0f;
    gettimeofday(&timeCallsReal, 0);
    timeCallsProcess = clock();
}


void setShaderUniforms()
{
    printf("SETTING THE SHADER UNIFORMS\n");
    glUseProgram(sProgram);
    glUniform1f(uMinSupportOffset, config.kernelMinFullSupport);
    glUniform1f(uWToMaxSupportRatio, (config.kernelMaxFullSupport-config.kernelMinFullSupport)/config.wProjectionMaxW); //(maxSuppor-minSupport) / maxW
    glUniform1f(uGridSize, config.gridDimension);
    glUniform1f(uWScale, config.wScale);
    glUniform1f(uUVScale, config.fieldOfView * 3.5);
    glUseProgram(0);
    
    glUseProgram(sProgramRender);
    glUniform1f(uGridSizeRender, config.gridDimension);
    glUseProgram(0);  
     printf("DONE WITH SETTING THE SHADER UNIFORMS\n");
}


void runGridder(void) {
    
    if(!config.visibilitiesFromFile)
    {
        for (int i = 0; i < config.visibilityCount * config.numVisibilityParams; i += config.numVisibilityParams) {

            visibilities[i] = 0.0f;//(float) (rand() % (int) config.gridDimension);
            visibilities[i + 1] = 0.0f;//(float) (rand() % (int) config.gridDimension);
            visibilities[i + 2] = 1000.0f;//(float) randomKernel;
            visibilities[i + 3] = 1.0f;//((float)rand()/RAND_MAX * 2.0f)-1.0f;
            visibilities[i + 4] = 1.0f;//((float)rand()/RAND_MAX * 2.0f)-1.0f;
        }
    }

    glBindFramebuffer(GL_FRAMEBUFFER, fboID);
    glEnable(GL_BLEND);
    glViewport(0, 0, config.gridDimension, config.gridDimension);

    struct timeval timeFunctionReal;
    gettimeofday(&timeFunctionReal, 0);
    int timeFunctionProcess = clock();

    // glPushAttrib(GL_VIEWPORT_BIT);

    glUseProgram(sProgram);

    glBlendEquationSeparate(GL_FUNC_ADD, GL_FUNC_ADD);
    glBlendFuncSeparate(GL_ONE, GL_ONE, GL_ONE, GL_ONE);
    // glBindBuffer(GL_ARRAY_BUFFER, guiRenderBoundsBuffer);

    glBindTexture(GL_TEXTURE_3D, kernalTextureID);
    glUniform1i(uShaderTextureKernalHandle, 0);


    //glBindBuffer(GL_ARRAY_BUFFER, guiRenderBoundsBuffer);
    glBindBuffer(GL_ARRAY_BUFFER, visibilityBuffer);
    glBufferData(GL_ARRAY_BUFFER, sizeof (GLfloat) * config.numVisibilityParams * config.visibilityCount, visibilities, GL_STATIC_DRAW);
    glEnableVertexAttribArray(sLocPosition);
    glVertexAttribPointer(sLocPosition, 3, GL_FLOAT, GL_FALSE, config.numVisibilityParams*sizeof(GLfloat), 0);
    glEnableVertexAttribArray(sComplex);
    glVertexAttribPointer(sComplex, 2, GL_FLOAT, GL_FALSE, config.numVisibilityParams*sizeof(GLfloat), (void*) (3*sizeof(GLfloat)));

//    int batchSize = config.visibilityCount/16;
//    int end = batchSize;
//    int start = 0;
//    for(int i=0;i<16;i++)
//    {
//        glDrawArrays(GL_POINTS, start, end);
//       // glFinish();
//        start += batchSize;
//        end +=batchSize;
//    }
    
    glDrawArrays(GL_POINTS, 0, config.visibilityCount);
     
    for (GLenum err = glGetError(); err != GL_NO_ERROR; err = glGetError()) {
        fprintf(stderr, "%d: %s\n", err, gluErrorString(err));
    }
    glDisableVertexAttribArray(sComplex);
    glDisableVertexAttribArray(sLocPosition);
    //glDisableVertexAttribArray(sLocColor);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glBindTexture(GL_TEXTURE_3D, 0);
    glUseProgram(0);
    
    glDisable(GL_BLEND);
    
    iterationCount++;
    
    
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
    //glPopAttrib();

        
   glFlush();
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
 
    if(iterationCount == config.displayDumpTime)
    {   glFlush();
        glFinish();
        glBindFramebuffer(GL_FRAMEBUFFER, fboID);
        iterationCount = 0;
        printGrid();
        glClearColor(0.0f, 0.0f, 0.0f, 0.0f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        glBindFramebuffer(GL_FRAMEBUFFER, 0);
    }
    glFlush();
    glFinish();
    
    glutSwapBuffers();
       printTimesAverage(timeFunctionReal,timeFunctionProcess,"ENTIRE FUNCTION TIME");
    gettimeofday(&timeCallsReal, 0);
    timeCallsProcess = clock();
     
}

void printGrid(void)
{
    glFinish();
    glReadPixels(0, 0, config.gridDimension, config.gridDimension, GL_RGBA, GL_FLOAT, gridBuffer);
    glFinish();
    int printSUM = 12;//config.gridDimension;//
//    //printf("Sampled Grid\n");
//    for(int row = 0; row < config.gridDimension; row++)
//    {
//        for(int col = 0; col < (config.gridDimension*4); col+=4)
//        {
//            float r = gridBuffer[(row*(int)config.gridDimension*4)+col];
//            float g = gridBuffer[(row*(int)config.gridDimension*4)+col+1];
////            float b = gridBuffer[(row*(int)config.gridDimension*4)+col+2];
////            float a = gridBuffer[(row*(int)config.gridDimension*4)+col+3];
//           // if(r > 0.001 || g > 0.001)
//               printf("(%.2f)",r);
//
//        }
//        printf("\n");
//    }
    printf("\nTransfered back completed!!!!!!!!\n");
}

void createKernel(int depth)
{   
    float start = -1.0 + (1.0/(float)config.kernelTexSize);
    float step = 2.0f/(float)config.kernelTexSize;
    
    float * curve = malloc(sizeof(float) * config.kernelTexSize);

    for(int i = 0; i < config.kernelTexSize; i++)
        curve[i] = start+(i*step);

    calculateSpheroidalCurve(curve, config.kernelTexSize);

    for(int row = 0; row < config.kernelTexSize; row++)
    {   
        for(int col = 0; col < config.kernelTexSize; col++)
        {
            kernelBuffer[(depth * config.kernelTexSize * config.kernelTexSize) + config.kernelTexSize * row + col].real = curve[row] * curve[col];
            kernelBuffer[(depth * config.kernelTexSize * config.kernelTexSize) + config.kernelTexSize * row + col].imaginary = (float) depth;
            //printf("%f ", kernelBuffer[(depth * config.kernelFullSupport * config.kernelFullSupport) + config.kernelFullSupport * row + col].real);
        }
        // printf("\n");
    }
    // printf("\n");
    free(curve);
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

void calculateSpheroidalCurve(float * nu, int kernelWidth)
{   
    float p[2][5] = {{8.203343e-2, -3.644705e-1, 6.278660e-1, -5.335581e-1, 2.312756e-1},
                     {4.028559e-3, -3.697768e-2, 1.021332e-1, -1.201436e-1, 6.412774e-2}};
    float q[2][3] = {{1.0000000e0, 8.212018e-1, 2.078043e-1},
                     {1.0000000e0, 9.599102e-1, 2.918724e-1}};
    
    int pNum = 5;
    int qNum = 3;
    
    for(int i = 0; i < kernelWidth; i++)
        nu[i] = fabsf(nu[i]);
    
    int part[kernelWidth];
    float nuend[kernelWidth];
    for(int i = 0; i < kernelWidth; i++)
    {
        if(nu[i] >= 0.0f && nu[i] <= 0.75f)
            part[i] = 0;
        else if(nu[i] > 0.75f && nu[i] < 1.0f)
            part[i] = 1;
        
        if(nu[i] >= 0.0f && nu[i] <= 0.75f)
            nuend[i] = 0.75f;
        else if(nu[i] > 0.75f && nu[i] < 1.0f)
            nuend[i] = 1.0f;      
    }
    
    float delnusq[kernelWidth];
    for(int i = 0; i < kernelWidth; i++)
        delnusq[i] = (nu[i] * nu[i]) - (nuend[i] * nuend[i]);
    
    float top[kernelWidth];
    for(int i = 0; i < kernelWidth; i++)
        top[i] = p[part[i]][0];
    
    for(int i = 1; i < pNum; i++)
        for(int y = 0; y < kernelWidth; y++)
            top[y] += (p[part[y]][i] * pow(delnusq[y], i)); 
    
    float bottom[kernelWidth];
    for(int i = 0; i < kernelWidth; i++)
        bottom[i] = q[part[i]][0];
    
    for(int i = 1; i < qNum; i++)
        for(int y = 0; y < kernelWidth; y++)
            bottom[y] += (q[part[y]][i] * pow(delnusq[y], i));
    
    for(int i = 0; i < kernelWidth; i++)
    {   
        float absNu = abs(nu[i]);
        nu[i] = (bottom[i] > 0.0f) ? top[i]/bottom[i] : 0.0f;
        if(absNu > 1.0f)
            nu[i] = 0.0f;
    }
}

double sumLegendreSeries(const double x, const int m)
{
    const int nOrders = ((int)spheroidal.itsCoeffs->size)*2 + (spheroidal.itsREven ? 0 : 1);
    double *vals = (double*) malloc(sizeof(double) * (nOrders+1));
    
    const int status = gsl_sf_legendre_sphPlm_array(nOrders+m, m, x, vals);
    double result = 0;
    
    for(unsigned int elem = 0; elem<spheroidal.itsCoeffs->size; ++elem)
    {
        const int r = 2*elem + (spheroidal.itsREven ? 0 : 1);
        result += gsl_vector_get(spheroidal.itsCoeffs, elem) * vals[r];
    }
    
    free(vals);
    
    if(status != GSL_SUCCESS)
    {
        printf("ERROR: Error calculating associated Legendre functions, status = %d", status);
    }
    
    return result;
}

void fillHelperMatrix(gsl_matrix *B, const int m)
{

    const double cSquared = config.prolateC*config.prolateC;
    
    for(unsigned int row = 0; row < B->size1; ++row) {
        const int r = ((int) row) * 2 + (spheroidal.itsREven ? 0 : 1);
        const int l = r + m; // order of Legendre function P_l^m
        double result = 0;
        
        result = ((double) l*(l+1)) + cSquared*(((double) 2*l+3)*(l+m)*(l-m)+((double) 2*l-1)*(l+m+1)*(l-m+1)) /
            (((double) 2*l+1)*(2*l-1)*(2*l+3));
        gsl_matrix_set(B, row, row, result);
        
        if (row>=1) 
        {
            result = cSquared/((double) 2*l-1)*sqrt(((double) l+m)*(l+m-1)*(l-m)*(l-m-1)/(((double) 2*l+1)*(2*l-3)));
            gsl_matrix_set(B, row, row-1, result);
        }
        if (row+1<B->size1) 
        {
            result = cSquared/((double) 2*l+3)*sqrt(((double) l+m+1)*(l+m+2)*(l-m+1)*(l-m+2)/
                             (((double) 2*l+1)*(2*l+5)));
            gsl_matrix_set(B, row, row+1, result);
        }
    }
}

double fillLegendreCoeffs(const gsl_matrix *B)
{
    gsl_vector *newCoeffs = gsl_vector_alloc(B->size1);
    gsl_vector_free(spheroidal.itsCoeffs);
    spheroidal.itsCoeffs = newCoeffs;
    
    gsl_matrix *A = gsl_matrix_alloc(B->size1, B->size1);
    gsl_matrix *eVec = gsl_matrix_alloc(B->size1, B->size1);
    gsl_eigen_symmv_workspace *work = gsl_eigen_symmv_alloc(B->size1);
    gsl_vector *eVal = gsl_vector_alloc(spheroidal.itsCoeffs->size);
    
    for(unsigned int row = 0; row < B->size1; ++row)
        for(unsigned int col = 0; col < B->size2; ++col)
            gsl_matrix_set(A, row, col, gsl_matrix_get(B, row, col));
    
    const int status = gsl_eigen_symmv(A, eVal, eVec, work);
    double result = -1;
    unsigned int optIndex = 0;
    
    if(status == GSL_SUCCESS)
    {
        for(unsigned int elem = 0; elem < spheroidal.itsCoeffs->size; ++elem)
        {
            const double val = gsl_vector_get(eVal, elem);
            if((elem == 0) || (val < result))
            {
                result = val;
                optIndex = elem;
            }
        }
    }
    
    for(size_t i = 0; i < B->size1; ++i)
        gsl_vector_set(spheroidal.itsCoeffs, i, gsl_matrix_get(eVec, i, optIndex));
    
    gsl_matrix_free(A);
    gsl_matrix_free(eVec);
    gsl_eigen_symmv_free(work);
    gsl_vector_free(eVal);
    
    return result;
}

void initSpheroidal(void)
{
    spheroidal.itsAlpha = config.prolateAlpha;
    spheroidal.itsREven = true;
    spheroidal.itsCoeffs = gsl_vector_alloc(config.prolateNumTerms);
    
    gsl_matrix *hlp = gsl_matrix_alloc(config.prolateNumTerms, config.prolateNumTerms);
    
    fillHelperMatrix(hlp, (int) config.prolateAlpha);
    
    fillLegendreCoeffs(hlp);
    
    spheroidal.itsSum0 = sumLegendreSeries(0, (int) config.prolateAlpha);
}

double calculateSpheroidalPoint(const double nu)
{    
    if(nu <= -1.0 || nu >= 1.0)
        return 0;
    else
    {
        const double res = sumLegendreSeries(nu, (int) spheroidal.itsAlpha) / spheroidal.itsSum0;
        return res * pow(1.0-nu*nu, -spheroidal.itsAlpha/2.0);
    }
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
         //toggle = 1;
        // usleep(5000000);
    }
}

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

GLuint createShader(GLenum shaderType, const char* shaderSource) {
    GLuint shader = glCreateShader(shaderType);
    glShaderSource(shader, 1, (const GLchar **) &shaderSource, NULL);
    glCompileShader(shader);
    checkShaderStatus(shader);
    return shader;
}

GLuint createProgram(GLuint fragmentShader, GLuint vertexShader) {
    GLuint program = glCreateProgram();
    glAttachShader(program, fragmentShader);
    glAttachShader(program, vertexShader);
    glLinkProgram(program);
    checkProgramStatus(program);
    return program;
}

void createWPlanes(void)
{       
    float maxW = 500.0f;
    float wStep = 100.0f;
    int numPlanes = 6;//(int) config.wProjectionMaxPlane + 1.0f;
    
    float start = -1.0 + (1.0/(float)config.kernelTexSize);
    float step = 2.0f/(float)config.kernelTexSize;
    
    float * curve = malloc(sizeof(float) * config.kernelTexSize);
    // Calculate steps
    for(int i = 0; i < config.kernelTexSize; i++)
        curve[i] = start+(i*step);
    // Calculate curve from steps
    calcSpheroidalCurve(curve);
    FloatComplex spheroidal2d[config.kernelTexSize][config.kernelTexSize];
    // Populate 2d complex spheroidal
    for(int r = 0; r < config.kernelTexSize; r++)
    {
        for(int c = 0; c < config.kernelTexSize; c++)
        {
            spheroidal2d[r][c].real = curve[r] * curve[c];
            spheroidal2d[r][c].imaginary = curve[r] * curve[c];
            //printf("%.3f ", spheroidal2d[r][c].real);
        }   
        //printf("\n");
    }
    
    free(curve);
    
    printf("createWPlanes: Max w: %.1f, step is %.1f wavelengths\n", maxW, wStep);
    
    float wList[numPlanes];
    float wIncrement = 0.0f;
    for(int i = 0; i < numPlanes; i++, wIncrement += wStep)
        wList[i] = wIncrement;
    
    // Dissolve 3d array into 2d array (kernelFullSupport x wProjectNumPlanes)
    FloatComplex wKernelList[numPlanes][(int)config.kernelTexSize][(int)config.kernelTexSize];
    
    for(int i = 0; i < numPlanes; i++)
    {
        // Create w screen
        FloatComplex wScreen[config.kernelTexSize][config.kernelTexSize];
        FloatComplex inverseShiftOutput[config.kernelTexSize][config.kernelTexSize];
        FloatComplex fftResult[config.kernelTexSize][config.kernelTexSize];
        FloatComplex shiftOutput[config.kernelTexSize][config.kernelTexSize];
        for(int r = 0; r < config.kernelTexSize; r++)
            for(int c = 0; c < config.kernelTexSize; c++)
            {
                wScreen[r][c].real = 0.0f;
                wScreen[r][c].imaginary = 0.0f;
                inverseShiftOutput[r][c].real = 0.0f;
                inverseShiftOutput[r][c].imaginary = 0.0f;
                fftResult[r][c].real = 0.0f;
                fftResult[r][c].imaginary = 0.0f;
                shiftOutput[r][c].real = 0.0f;
                shiftOutput[r][c].imaginary = 0.0f;
            }
        
        createWTermLike(config.gridDimension, wScreen, wList[i]);
        
        for(int r = 0; r < config.kernelTexSize; r++)
            for(int c = 0; c < config.kernelTexSize; c++)
            {
                // Complex multiplication
                wScreen[r][c] = complexMultiply(wScreen[r][c], spheroidal2d[r][c]);
            }
        
        // Inverse shift FFT for transformation
        fft2dInverseShift(config.kernelTexSize, wScreen, inverseShiftOutput);
        
        // FFT image
        fft2dVectorRadixTransform(config.kernelTexSize, inverseShiftOutput, fftResult);
        
        // Inverse shift FFT'd result
        fft2dShift(config.kernelTexSize, fftResult, shiftOutput);
        
        // Append to list of kernels
        for(int r = 0; r < config.kernelTexSize; r++)
        {
            for(int c = 0; c < config.kernelTexSize; c++)
            {
                wKernelList[i][r][c] = shiftOutput[r][c];
                printf("%20.3f ", shiftOutput[r][c].imaginary);
            }
             printf("\n");   
        }
        printf("\n\n");
    }
    
    // Copy to kernel buffer global
    
}

void createWTermLike(int width, FloatComplex wScreen[][width], float w)
{    
    float fieldOfView = 0.001f * config.gridDimension;
    
    float fresnel = fabsf(w) * ((0.5 * fieldOfView)*(0.5 * fieldOfView));
    printf("CreateWTermLike: For w = %f, field of view = %f, fresnel number = %f\n", w, fieldOfView, fresnel);
    wBeam(width, wScreen, w, width/2, width/2, fieldOfView);
}

void wBeam(int width, FloatComplex wScreen[][width], float w, float centerX, float centerY, float fieldOfView)
{
    float r2[width][width];
    float ph[width][width];
    
    for(int r = 0; r < width; r++)
        for(int c = 0; c < width; c++)
        {
            float l = ((r-centerY) / width)*fieldOfView;
            float m = ((c-centerX) / width)*fieldOfView;
            r2[r][c] = (l*l)+(m*m);
            
            if(r2[r][c] < 1.0f)
                ph[r][c] = w * (1.0 - sqrtf(1.0 - r2[r][c]));
            else
                ph[r][c] = 0.0f;
        }

    for(int r = 0; r < width; r++)
    {
        for(int c = 0; c < width; c++)
        {
            if(r2[r][c] < 1.0f)
            {
                wScreen[r][c] = complexExponential(ph[r][c]);   
            }
            else if(r2[r][c] == 0.0f)
                wScreen[r][c] = (FloatComplex) {.real = 1.0f, .imaginary = 0.0f};
            else
                wScreen[r][c] = (FloatComplex) {.real = 0.0f, .imaginary = 0.0f};
        }        
    }
}

void fft2dVectorRadixTransform(int numChannels, const FloatComplex input[][numChannels], FloatComplex output[][numChannels])
{   
    // Calculate bit reversed indices
    int* bitReversedIndices = malloc(sizeof(int) * numChannels);
    calcBitReversedIndices(numChannels, bitReversedIndices);
    
    // Copy data to result for processing
    for(int r = 0; r < numChannels; r++)
        for(int c = 0; c < numChannels; c++)
            output[r][c] = input[bitReversedIndices[r]][bitReversedIndices[c]];
    free(bitReversedIndices);
    
    // Use butterfly operations on result to find the DFT of original data
    for(int m = 2; m <= numChannels; m *= 2)
    {
        FloatComplex omegaM = (FloatComplex) {.real = cosf(M_PI * 2.0 / m), .imaginary = -sinf(M_PI * 2.0 / m)};
        
        for(int k = 0; k < numChannels; k += m)
        {
            for(int l = 0; l < numChannels; l += m)
            {
                FloatComplex x = (FloatComplex) {.real = 1.0f, .imaginary = 0.0f};
                
                for(int i = 0; i < m / 2; i++)
                {
                    FloatComplex y = (FloatComplex) {.real = 1.0f, .imaginary = 0.0f};
                    
                    for(int j = 0; j < m / 2; j++)
                    {
                        // Perform 2D butterfly operation in-place at (k+j, l+j)
                        FloatComplex in00 = output[k+i][l+j];
                        FloatComplex in01 = complexMultiply(output[k+i][l+j+m/2], y);
                        FloatComplex in10 = complexMultiply(output[k+i+m/2][l+j], x);
                        FloatComplex in11 = complexMultiply(complexMultiply(output[k+i+m/2][l+j+m/2], x), y);
                        
                        FloatComplex temp00 = complexAdd(in00, in01);
                        FloatComplex temp01 = complexSubtract(in00, in01);
                        FloatComplex temp10 = complexAdd(in10, in11);
                        FloatComplex temp11 = complexSubtract(in10, in11);
                        
                        output[k+i][l+j] = complexAdd(temp00, temp10);
                        output[k+i][l+j+m/2] = complexAdd(temp01, temp11);
                        output[k+i+m/2][l+j] = complexSubtract(temp00, temp10);
                        output[k+i+m/2][l+j+m/2] = complexSubtract(temp01, temp11);
                        y = complexMultiply(y, omegaM);
                    }
                    x = complexMultiply(x, omegaM);
                }
            }
        }
    }
}

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

void fft2dShift(int numChannels, FloatComplex input[][numChannels], FloatComplex shifted[][numChannels]) 
{
    for(int i = 0; i < numChannels; i++)
    {
        for(int j = 0; j < numChannels; j++)
        {            
            // Top-left
            if(i < numChannels/2 && j < numChannels/2)
                shifted[i+numChannels/2][j+numChannels/2] = input[i][j];
            // Bottom-left
            else if(i >= numChannels/2 && j < numChannels/2)
                shifted[i-numChannels/2][j+numChannels/2] = input[i][j];
            // Top-right
            else if(i < numChannels/2 && j >= numChannels/2)
                shifted[i+numChannels/2][j-numChannels/2] = input[i][j];
            // Bottom-right
            else
                shifted[i-numChannels/2][j-numChannels/2] = input[i][j];
        }
    }
}

void fft2dInverseShift(int numChannels, FloatComplex input[][numChannels], FloatComplex inverse[][numChannels]) 
{
    for(int i = 0; i < numChannels; i++)
    {
        for(int j = 0; j < numChannels; j++)
        {            
            // Top-left
            if(i < numChannels/2 && j < numChannels/2)
                inverse[i+numChannels/2][j+numChannels/2] = input[i][j];
            // Bottom-left
            else if(i >= numChannels/2 && j < numChannels/2)
                inverse[i-numChannels/2][j+numChannels/2] = input[i][j];
            // Top-right
            else if(i < numChannels/2 && j >= numChannels/2)
                inverse[i+numChannels/2][j-numChannels/2] = input[i][j];
            // Bottom-right
            else
                inverse[i-numChannels/2][j-numChannels/2] = input[i][j];
        }
    }
}

FloatComplex complexAdd(FloatComplex x, FloatComplex y)
{
    FloatComplex z;
    z.real = x.real + y.real;
    z.imaginary = x.imaginary + y.imaginary;
    return z;
}

FloatComplex complexSubtract(FloatComplex x, FloatComplex y)
{
    FloatComplex z;
    z.real = x.real - y.real;
    z.imaginary = x.imaginary - y.imaginary;
    return z;
}

FloatComplex complexMultiply(FloatComplex x, FloatComplex y)
{
    FloatComplex z;
    z.real = x.real*y.real - x.imaginary*y.imaginary;
    z.imaginary = x.imaginary*y.real + x.real*y.imaginary;
    return z;
}

FloatComplex complexExponential(float ph)
{
    return (FloatComplex) {.real = cosf(2*M_PI*ph), .imaginary = -sinf(2*M_PI*ph)};
}

// Nu should be array of points between -1 && -1
void calcSpheroidalCurve(float * curve)
{   
    float p[2][5] = {{8.203343e-2, -3.644705e-1, 6.278660e-1, -5.335581e-1, 2.312756e-1},
                     {4.028559e-3, -3.697768e-2, 1.021332e-1, -1.201436e-1, 6.412774e-2}};
    float q[2][3] = {{1.0000000e0, 8.212018e-1, 2.078043e-1},
                     {1.0000000e0, 9.599102e-1, 2.918724e-1}};
    
    int pNum = 5;
    int qNum = 3;
    
    for(int i = 0; i < config.kernelTexSize; i++)
        curve[i] = fabsf(curve[i]);
    
    int part[config.kernelTexSize];
    float nuend[config.kernelTexSize];
    for(int i = 0; i < config.kernelTexSize; i++)
    {
        if(curve[i] >= 0.0f && curve[i] <= 0.75f)
            part[i] = 0;
        else if(curve[i] > 0.75f && curve[i] < 1.0f)
            part[i] = 1;
        
        if(curve[i] >= 0.0f && curve[i] <= 0.75f)
            nuend[i] = 0.75f;
        else if(curve[i] > 0.75f && curve[i] < 1.0f)
            nuend[i] = 1.0f;      
    }
    
    float delnusq[config.kernelTexSize];
    for(int i = 0; i < config.kernelTexSize; i++)
        delnusq[i] = (curve[i] * curve[i]) - (nuend[i] * nuend[i]);
    
    float top[config.kernelTexSize];
    for(int i = 0; i < config.kernelTexSize; i++)
        top[i] = p[part[i]][0];
    
    for(int i = 1; i < pNum; i++)
        for(int y = 0; y < config.kernelTexSize; y++)
            top[y] += (p[part[y]][i] * pow(delnusq[y], i)); 
    
    float bottom[config.kernelTexSize];
    for(int i = 0; i < config.kernelTexSize; i++)
        bottom[i] = q[part[i]][0];
    
    for(int i = 1; i < qNum; i++)
        for(int y = 0; y < config.kernelTexSize; y++)
            bottom[y] += (q[part[y]][i] * pow(delnusq[y], i));
    
    for(int i = 0; i < config.kernelTexSize; i++)
    {   
        float absCurve = abs(curve[i]);
        curve[i] = (bottom[i] > 0.0f) ? top[i]/bottom[i] : 0.0f;
        if(absCurve > 1.0f)
            curve[i] = 0.0f;
    }
}

