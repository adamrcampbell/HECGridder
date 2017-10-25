
/* 
 * File:   gridder.c
 * Author: adam
 *
 * Created on 1 August 2017, 11:23 AM
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
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_vector.h>
#include <gsl/gsl_linalg.h>
#include <gsl/gsl_eigen.h>
#include <gsl/gsl_sf.h>     

#include "gridder.h"
#include "gpu.h"

/*--------------------------------------------------------------------
 *   GLOBAL CONFIG
 *-------------------------------------------------------------------*/
static float gridDimension;
static int kernelSize;
static enum kernel kernelType;
static int visibilityCount;
static int visibilityParams;

/*--------------------------------------------------------------------
 *   GUI CONFIG
 *-------------------------------------------------------------------*/
static int refreshDelay;
static GLfloat guiRenderBounds[8];

/*--------------------------------------------------------------------
 *   KAISER BESSEL CONFIG
 *-------------------------------------------------------------------*/
static float kaiserAlpha;
static float accuracy;
static float piAlpha;

/*--------------------------------------------------------------------
 *   PROLATE SPHEROIDAL CONFIG
 *-------------------------------------------------------------------*/
static double prolateC;
static double prolateAlpha;
static unsigned int numTerms;
static struct SpheroidalFunction spheroidal;

/*--------------------------------------------------------------------
 *   W-PROJECTION CONFIG
 *-------------------------------------------------------------------*/
static float kernelStart;
static float kernelStep;
static float wStep;
static float wMaxAbs;
static float cellSize;
static float fieldOfView;

static GLuint sProgram;
static GLuint sLocPosition;
static GLuint sComplex;
static GLuint sLocColor;
static GLuint guiRenderBoundsBuffer;
static GLuint visibilityBuffer;

static GLuint sLocPositionRender;
static GLuint sProgramRender;
static GLuint uShaderTextureHandle;
static GLuint uShaderTextureKernalHandle;

static GLuint fboID;
static GLuint textureID;
static GLuint kernalTextureID;

static GLfloat* gridBuffer;
static FloatComplex* kernelBuffer;
static GLuint* visibilityIndices;
static GLfloat* visibilities;

int iterationCount = 0;
const int dumpTime = 1;

int counter =0;
int counterAverage;
double sumTimeReal;
float sumTimeProcess;
int val;
struct timeval timeCallsReal;
int timeCallsProcess;
bool toggle = 0;

const int windowDisplay = 800;

void initConfig(void) {
    // Global
    gridDimension = 130.0f;
    kernelSize = 128;
    kernelType = PROLATE;
    visibilityCount = 1;
    visibilityParams = 5;

    // Gui
    refreshDelay = 500;
    GLfloat renderTemp[8] = {
        0.0f, 0.0f,
        0.0f, gridDimension,
        gridDimension, 0.0f,
        gridDimension, gridDimension
    };
    memcpy(guiRenderBounds, renderTemp, sizeof (guiRenderBounds));
    
    // Kaiser
    kaiserAlpha = 2.59f;
    accuracy = 1E-12;
    piAlpha = M_PI * kaiserAlpha;
    
    // Prolate
    prolateC = 3 * M_PI;
    prolateAlpha = 1;
    numTerms = 16; // default: 16
    
    // W-Projection
    kernelStart = (-1.0 + (1.0/(float)kernelSize));
    kernelStep = (2.0f/(float)kernelSize);
    wStep = 1000.0f;
    wMaxAbs = 25000.0f;
    cellSize = 0.01f;
    fieldOfView = cellSize * (float) kernelSize; 
}

void initGridder(void) {
   
    kernelBuffer = malloc(sizeof (FloatComplex) * kernelSize * kernelSize * 3);
    if(kernelType == PROLATE)
        initSpheroidal();
    
    glClampColorARB(GL_CLAMP_VERTEX_COLOR_ARB, GL_FALSE);
    glClampColorARB(GL_CLAMP_READ_COLOR_ARB, GL_FALSE);
    glClampColorARB(GL_CLAMP_FRAGMENT_COLOR_ARB, GL_FALSE);

    srand(time(NULL));
    int size = 4 * gridDimension*gridDimension;
    gridBuffer = (GLfloat*) malloc(sizeof (GLfloat) * size);

    for (int i = 0; i < size; i += 4) {
        gridBuffer[i] = 0.0f;
        gridBuffer[i + 1] = 0.0f;
        gridBuffer[i + 2] = 0.0f;
        gridBuffer[i + 3] = 0.0f;
    }

    visibilities = malloc(sizeof (GLfloat) * visibilityParams * visibilityCount);
    visibilityIndices = malloc(sizeof (GLuint) * visibilityCount);

    for (GLuint i = 0; i < visibilityCount; i++) {
        visibilityIndices[i] = i;
    }
    
    char buffer[2000];
    sprintf(buffer, VERTEX_SHADER, gridDimension);
    GLuint vertexShader = createShader(GL_VERTEX_SHADER, buffer);
    GLuint fragmentShader = createShader(GL_FRAGMENT_SHADER, FRAGMENT_SHADER);
    sProgram = createProgram(vertexShader, fragmentShader);
    sLocPosition = glGetAttribLocation(sProgram, "position");
    sComplex = glGetAttribLocation(sProgram, "complex");
    uShaderTextureKernalHandle = glGetUniformLocation(sProgram, "kernalTex");
    
    sprintf(buffer, VERTEX_SHADER_RENDER, gridDimension);
    vertexShader = createShader(GL_VERTEX_SHADER, buffer);
    fragmentShader = createShader(GL_FRAGMENT_SHADER, FRAGMENT_SHADER_RENDER);
    sProgramRender = createProgram(vertexShader, fragmentShader);
    sLocPositionRender = glGetAttribLocation(sProgramRender, "position");
    uShaderTextureHandle = glGetUniformLocation(sProgramRender, "destTex");
    
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
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, gridDimension, gridDimension, 0, GL_RGBA, GL_FLOAT, gridBuffer);

    glGenFramebuffers(1, idArray);
    fboID = idArray[0];
    glBindFramebuffer(GL_FRAMEBUFFER, fboID);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, textureID, 0);
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
    glBindTexture(GL_TEXTURE_2D, 0);
    
    for(int i = 0; i < 3; i++)
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
    glTexImage3D(GL_TEXTURE_3D, 0,  GL_RG32F, kernelSize, kernelSize, 3, 0, GL_RG, GL_FLOAT, kernelBuffer);
    glBindTexture(GL_TEXTURE_3D, 0);
    
    glTexEnvi(GL_POINT_SPRITE, GL_COORD_REPLACE, GL_TRUE);
    glEnable(GL_POINT_SPRITE);
    glEnable(GL_VERTEX_PROGRAM_POINT_SIZE);
    
    counter = 0;
    sumTimeProcess = 0.0f;
    gettimeofday(&timeCallsReal, 0);
    timeCallsProcess = clock();
}

void runGridder(void) {
    
    for (int i = 0; i < visibilityCount * visibilityParams; i += visibilityParams) {
        
//        int randomKernel = (int)((float)rand()/RAND_MAX * 121)+7;
//        
//        while(randomKernel % 2 == 0)
//        {
//            randomKernel = (int)((float)rand()/RAND_MAX * 121)+7;
//        }
        
        visibilities[i] = (float) 63;//(rand() % (int) gridDimension);
        visibilities[i + 1] = (float) 63;// (rand() % (int) gridDimension);
        visibilities[i + 2] = (float) kernelSize;
        visibilities[i + 3] = ((float)rand()/RAND_MAX * 2.0f)-1.0f;
        visibilities[i + 4] = ((float)rand()/RAND_MAX * 2.0f)-1.0f;
    }

    glBindFramebuffer(GL_FRAMEBUFFER, fboID);
    glEnable(GL_BLEND);
    glViewport(0, 0, gridDimension, gridDimension);

    struct timeval timeFunctionReal;
    gettimeofday(&timeFunctionReal, 0);
    int timeFunctionProcess = clock();
    struct timeval timePartReal;
    int timePartProcess = 0;

    // glPushAttrib(GL_VIEWPORT_BIT);

    glUseProgram(sProgram);

    glBlendEquationSeparate(GL_FUNC_ADD, GL_FUNC_ADD);
    glBlendFuncSeparate(GL_ONE, GL_ONE, GL_ONE, GL_ONE);
    // glBindBuffer(GL_ARRAY_BUFFER, guiRenderBoundsBuffer);

    glBindTexture(GL_TEXTURE_3D, kernalTextureID);
    glUniform1i(uShaderTextureKernalHandle, 0);


    //glBindBuffer(GL_ARRAY_BUFFER, guiRenderBoundsBuffer);
    glBindBuffer(GL_ARRAY_BUFFER, visibilityBuffer);
    glBufferData(GL_ARRAY_BUFFER, sizeof (GLfloat) * visibilityParams * visibilityCount, visibilities, GL_STATIC_DRAW);
    glEnableVertexAttribArray(sLocPosition);
    glVertexAttribPointer(sLocPosition, 3, GL_FLOAT, GL_FALSE, visibilityParams*sizeof(GLfloat), 0);
    glEnableVertexAttribArray(sComplex);
    glVertexAttribPointer(sComplex, 2, GL_FLOAT, GL_FALSE, visibilityParams*sizeof(GLfloat), (void*) (3*sizeof(GLfloat)));

    glDrawElements(GL_POINTS, visibilityCount, GL_UNSIGNED_INT, visibilityIndices);

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
    
    if(iterationCount == dumpTime)
    {
        iterationCount = 0;
        printGrid();
        glClearColor(0.0f, 0.0f, 0.0f, 0.0f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    }
    
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
    //glPopAttrib();

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
    
    glutSwapBuffers();
    
    printTimesAverage(timeFunctionReal,timeFunctionProcess,"ENTIRE FUNCTION TIME");
    gettimeofday(&timeCallsReal, 0);
    timeCallsProcess = clock();
}

void printGrid(void)
{
    glFinish();
    glReadPixels(0, 0, gridDimension, gridDimension, GL_RGBA, GL_FLOAT, gridBuffer);

    printf("Sampled Grid\n");
    for(int row = 0; row < gridDimension; row++)
    {
        for(int col = 0; col < ((gridDimension)*4); col+=4)
        {
            float r = gridBuffer[(row*(int)gridDimension*4)+col];
            float g = gridBuffer[(row*(int)gridDimension*4)+col+1];
            float b = gridBuffer[(row*(int)gridDimension*4)+col+2];
            float a = gridBuffer[(row*(int)gridDimension*4)+col+3];

            if(row == 63)
                printf("%f\n", r);

        }
        //printf("\n");
    }
    printf("\n");
}

void createKernel(int depth)
{   
    float start = -1.0 + (1.0/(float)kernelSize);
    float step = 2.0f/(float)kernelSize;
    
    if(kernelType == KAISER)
    {
        for (int row = 0; row < kernelSize; row++) {
            
            float rowKernelWeight = calculateKernelWeight(start+(row*step));
            
            for (int col = 0, c = 0; col < kernelSize; col++, c++) {
                
                kernelBuffer[(depth * kernelSize * kernelSize) + kernelSize * row + col].real = rowKernelWeight * calculateKernelWeight(start+(col*step));
                kernelBuffer[(depth * kernelSize * kernelSize) + kernelSize * row + col].imaginary = 1.0f;
                // printf("%f ", kernelBuffer[(int) kernelSize * row + col]);
            }
            // printf("\n");
        }
    }
    else if(kernelType == PROLATE)
    {
        float * curve = malloc(sizeof(float) * kernelSize);
        
        for(int i = 0; i < kernelSize; i++)
            curve[i] = start+(i*step);
        
        calculateSpheroidalCurve(curve, kernelSize);
        
        for(int row = 0; row < kernelSize; row++)
        {
            for(int col = 0; col < kernelSize; col++)
            {
                kernelBuffer[(depth * kernelSize * kernelSize) + kernelSize * row + col].real = curve[row] * curve[col];
                kernelBuffer[(depth * kernelSize * kernelSize) + kernelSize * row + col].imaginary = (float) depth;
                //printf("%f ", kernelBuffer[(int) kernelSize * row + col].real);
            }
            //printf("\n");
        }
        free(curve);
    }
    else
        printf("ERROR: Unsupported kernel type\n");
}

double calculateKernelWeight(float x)
{   
    if(kernelType == KAISER)
        return calculateKaiserPoint(x);
    else if(kernelType == PROLATE)
        return calculateSpheroidalPoint(x);
}

int main(int argc, char** argv) {

    initConfig();
    
    srand((unsigned int) time(NULL));
    setenv("DISPLAY", ":0", 11.0);
    glutInit(&argc, argv);
    glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE | GLUT_ALPHA);
    glutInitWindowSize(windowDisplay, windowDisplay);
    glutInitWindowPosition(0, 0);
    glutCreateWindow("Derp Gridder");
    glutDisplayFunc(runGridder);
    glutTimerFunc(refreshDelay, timerEvent, 0);
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

float calculateKaiserPoint(float i) {
    if (i <= -1. || i >= 1.)
        return 0;

    float innerTerm = 2.0 * (i + 1.0) / 2 - 1;
    float x = piAlpha * sqrt(1 - innerTerm * innerTerm);
    float temp = getZeroOrderModifiedBessel(x) / getZeroOrderModifiedBessel(piAlpha);
    return temp;
}

float getZeroOrderModifiedBessel(float x) {
    float sum = 0;
    float term = 1;
    int m = 0;

    do {
        term = 1.0;

        for (int i = 1; i <= m; i++)
            term *= x / (2.0 * i);

        term *= term;
        sum += term;
        m++;
    } while (fabsf(term / sum) > accuracy);

    return sum;
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

    const double cSquared = prolateC*prolateC;
    
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
    spheroidal.itsAlpha = prolateAlpha;
    spheroidal.itsREven = true;
    spheroidal.itsCoeffs = gsl_vector_alloc(numTerms);
    
    gsl_matrix *hlp = gsl_matrix_alloc(numTerms, numTerms);
    
    fillHelperMatrix(hlp, (int) prolateAlpha);
    
    fillLegendreCoeffs(hlp);
    
    spheroidal.itsSum0 = sumLegendreSeries(0, (int) prolateAlpha);
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
    glutTimerFunc(refreshDelay, timerEvent, 0);
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

//void wKernelList(void)
//{   
//    float * curve = malloc(sizeof(float) * KERNEL_WIDTH);
//    // Calculate steps
//    for(int i = 0; i < KERNEL_WIDTH; i++)
//        curve[i] = KERNEL_START+(i*KERNEL_STEP);
//    // Calculate curve from steps
//    calcSpheroidalCurve(curve);
//    FloatComplex spheroidal2d[KERNEL_WIDTH][KERNEL_WIDTH];
//    // Populate 2d complex spheroidal
//    for(int r = 0; r < KERNEL_WIDTH; r++)
//        for(int c = 0; c < KERNEL_WIDTH; c++)
//        {
//            spheroidal2d[r][c].real = curve[r] * curve[c];
//            spheroidal2d[r][c].imaginary = curve[r] * curve[c];
//        }
//    
//    free(curve);
//    
//    printf("wKernelList: Max abs w: %.1f, step is %.1f wavelengths\n", W_MAX_ABS, W_STEP);
//    
//    int numWSteps = digitize(W_MAX_ABS*2, W_STEP);
//    float wList[numWSteps];
//    float wIncrement = -W_MAX_ABS;
//    for(int i = 0; i < numWSteps; i++, wIncrement += W_STEP)
//        wList[i] = wIncrement;
//    
//    FloatComplex wTemplate[KERNEL_WIDTH][KERNEL_WIDTH];
//    for(int r = 0; r < KERNEL_WIDTH; r++)
//        for(int c = 0; c < KERNEL_WIDTH; c++)
//        {
//            wTemplate[r][c].real = 0.0f;
//            wTemplate[r][c].imaginary = 0.0f;
//        }
//    
//    // 3d storage of padded 2d kernels
//    FloatComplex kernels[numWSteps][KERNEL_WIDTH][KERNEL_WIDTH];
//    
//    for(int i = 0; i < numWSteps; i++)
//    {
//        // Create w screen
//        FloatComplex wScreen[KERNEL_WIDTH][KERNEL_WIDTH];
//        for(int r = 0; r < KERNEL_WIDTH; r++)
//            for(int c = 0; c < KERNEL_WIDTH; c++)
//            {
//                wScreen[r][c].real = 0.0f;
//                wScreen[r][c].imaginary = 0.0f;
//            }
//        
//        createWTermLike(wScreen, wList[i]);
//        
//        for(int r = 0; r < KERNEL_WIDTH; r++)
//            for(int c = 0; c < KERNEL_WIDTH; c++)
//            {
//                // Complex multiplication
//                wScreen[r][c] *= spheroidal2d[r][c];
//            }
//        
//        // FFT image
//        FloatComplex result[KERNEL_WIDTH][KERNEL_WIDTH];
//        fft2DVectorRadixTransform(KERNEL_WIDTH, wScreen, result);
//        
//        // Append to list of kernels
//        for(int r = 0; r < KERNEL_WIDTH; r++)
//            for(int c = 0; c < KERNEL_WIDTH; c++)
//                kernels[i][r][c] = result[r][c];
//    }
//}
//
//void createWTermLike(int width, FloatComplex wScreen[][width], float w)
//{    
//    float fresnel = fabsf(w) * ((0.5 * FIELD_OF_VIEW)*(0.5 * FIELD_OF_VIEW));
//    printf("CreateWTermLike: For w = %f, field of view = %f, fresnel number = %f\n", w, FIELD_OF_VIEW, fresnel);
//    wBeam(wScreen, width, FIELD_OF_VIEW, w, width/2, width/2);
//}
//
//void wBeam(int width, FloatComplex wScreen[][width], int numPixel, 
//        float fieldOfView, float w, float centerX, float centerY)
//{
//    float r2[numPixel][numPixel];
//    float ph[numPixel][numPixel];
//    
//    for(int r = 0; r < numPixel; r++)
//        for(int c = 0; c < numPixel; c++)
//        {
//            float l = ((r-centerY) / numPixel)*fieldOfView;
//            float m = ((c-centerX) / numPixel)*fieldOfView;
//            r2[r][c] = (l*l)+(m*m);
//            
//            if(r2[r][c] < 1.0f)
//                ph[r][c] = w * (1.0 - sqrtf(1.0 - r2[r][c]));
//            else
//                ph[r][c] = 0.0f;
//        }
//
//    for(int r = 0; r < numPixel; r++)
//    {
//        for(int c = 0; c < numPixel; c++)
//        {
//            if(r2[r][c] < 1.0f)
//            {
//                // TODO
//                wScreen[r][c] = cexp(-2*I * PI * ph[r][c]);
//            }
//            else if(r2[r][c] == 0.0f)
//            {
//                wScreen[r][c].real = 1.0f;
//                wScreen[r][c].imaginary = 0.0f;
//            }
//            else
//            {
//                wScreen[r][c].real = 0.0f;
//                wScreen[r][c].imaginary = 0.0f;
//            }
//        }        
//    }
//}
//
//int digitize(float w, float wmaxabs)
//{
//    return (int) ceilf((w+wmaxabs)/W_STEP);
//}
//
//void fft2DVectorRadixTransform(int numChannels, const FloatComplex input[][numChannels], FloatComplex output[][numChannels])
//{   
//    // Calculate bit reversed indices
//    int* bitReversedIndices = calcBitReversedIndices(numChannels);
//    
//    // Copy data to result for processing
//    for(int r = 0; r < numChannels; r++)
//        for(int c = 0; c < numChannels; c++)
//            output[r][c] = input[bitReversedIndices[r]][bitReversedIndices[c]];
//    free(bitReversedIndices);
//    
//    // Use butterfly operations on result to find the DFT of original data
//    for(int m = 2; m <= numChannels; m *= 2)
//    {
//        float complex omegaM = CMPLX(cosf(PI * 2.0 / m), -sinf(PI * 2.0 / m));
//        for(int k = 0; k < numChannels; k += m)
//        {
//            for(int l = 0; l < numChannels; l += m)
//            {
//                float complex x = CMPLX(1.0, 0.0);
//                for(int i = 0; i < m / 2; i++)
//                {
//                    float complex y = CMPLX(1.0, 0.0);
//                    for(int j = 0; j < m / 2; j++)
//                    {
//                        // Perform 2D butterfly operation in-place at (k+j, l+j)
//                        float complex in00 = output[k+i][l+j];
//                        float complex in01 = output[k+i][l+j+m/2] * y;
//                        float complex in10 = output[k+i+m/2][l+j] * x;
//                        float complex in11 = (output[k+i+m/2][l+j+m/2] * x) * y;
//                        
//                        float complex temp00 = in00 + in01;
//                        float complex temp01 = in00 - in01;
//                        float complex temp10 = in10 + in11;
//                        float complex temp11 = in10 - in11;
//                        
//                        output[k+i][l+j] = temp00 + temp10;
//                        output[k+i][l+j+m/2] = temp01 + temp11;
//                        output[k+i+m/2][l+j] = temp00 - temp10;
//                        output[k+i+m/2][l+j+m/2] = temp01 - temp11;
//                        y *= omegaM;
//                    }
//                    x *= omegaM;
//                }
//            }
//        }
//    }
//}
//
//int* calcBitReversedIndices(int n)
//{
//    int* indices = malloc(n * sizeof(int));
//    
//    for(int i = 0; i < n; i++)
//    {
//        // Calculate index r to which i will be moved
//        unsigned int iPrime = i;
//        int r = 0;
//        for(int j = 1; j < n; j*=2)
//        {
//            int b = iPrime & 1;
//            r = (r << 1) + b;
//            iPrime = (iPrime >> 1);
//        }
//        indices[i] = r;
//    }
//    
//    return indices;
//}
//
//FloatComplex complexAdd(FloatComplex x, FloatComplex y)
//{
//    FloatComplex sum;
//    sum.real = x.real + y.real;
//    sum.imaginary = x.imaginary + y.imaginary;
//    return sum;
//}
//
//FloatComplex complexSubtract(FloatComplex x, FloatComplex y)
//{
//    FloatComplex diff;
//    diff.real = x.real - y.real;
//    diff.imaginary = x.imaginary - y.imaginary;
//    return diff;
//}
//
//FloatComplex complexDivide(FloatComplex x, FloatComplex y)
//{
//    
//}
//
//FloatComplex complexExponential(FloatComplex x)
//{
//    
//}

// Nu should be array of points between -1 && -1
void calcSpheroidalCurve(float * curve, int width)
{   
    float p[2][5] = {{8.203343e-2, -3.644705e-1, 6.278660e-1, -5.335581e-1, 2.312756e-1},
                     {4.028559e-3, -3.697768e-2, 1.021332e-1, -1.201436e-1, 6.412774e-2}};
    float q[2][3] = {{1.0000000e0, 8.212018e-1, 2.078043e-1},
                     {1.0000000e0, 9.599102e-1, 2.918724e-1}};
    
    int pNum = 5;
    int qNum = 3;
    
    for(int i = 0; i < width; i++)
        curve[i] = fabsf(curve[i]);
    
    int part[width];
    float nuend[width];
    for(int i = 0; i < width; i++)
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
    
    float delnusq[width];
    for(int i = 0; i < width; i++)
        delnusq[i] = (curve[i] * curve[i]) - (nuend[i] * nuend[i]);
    
    float top[width];
    for(int i = 0; i < width; i++)
        top[i] = p[part[i]][0];
    
    for(int i = 1; i < pNum; i++)
        for(int y = 0; y < width; y++)
            top[y] += (p[part[y]][i] * pow(delnusq[y], i)); 
    
    float bottom[width];
    for(int i = 0; i < width; i++)
        bottom[i] = q[part[i]][0];
    
    for(int i = 1; i < qNum; i++)
        for(int y = 0; y < width; y++)
            bottom[y] += (q[part[y]][i] * pow(delnusq[y], i));
    
    for(int i = 0; i < width; i++)
    {   
        float absCurve = abs(curve[i]);
        curve[i] = (bottom[i] > 0.0f) ? top[i]/bottom[i] : 0.0f;
        if(absCurve > 1.0f)
            curve[i] = 0.0f;
    }
}

void populate3DKernel(void)
{
    for(int depth = 0; depth < 3; depth++)
    {
        for(int row = 0; row < kernelSize; row++)
        {
            for(int col = 0; col < kernelSize; col++)
            {
                kernelBuffer[(depth * kernelSize * kernelSize) + kernelSize * row + col].real = (float) depth+1;
                kernelBuffer[(depth * kernelSize * kernelSize) + kernelSize * row + col].imaginary = 1.0f;
            }
        }
    }
}

