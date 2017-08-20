
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
static float kernelSize;
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
static GLfloat* kernalBuffer;
static GLuint* visibilityIndices;
static GLfloat* visibilities;

int iterationCount = 0;
const int dumpTime = 10;

int counter =0;
int counterAverage;
double sumTimeReal;
float sumTimeProcess;
int val;
struct timeval timeCallsReal;
int timeCallsProcess;
bool toggle = 0;

void initConfig(void) {
    // Global
    gridDimension = 1000.0f;
    kernelSize = 127.0f;
    kernelType = KAISER;
    visibilityCount = 19800;
    visibilityParams = 5;

    // Gui
    refreshDelay = 0;
    GLfloat renderTemp[8] = {
        0.0f, 0.0f,
        0.0f, gridDimension,
        gridDimension, 0.0f,
        gridDimension, gridDimension
    };
    memcpy(guiRenderBounds, renderTemp, sizeof (guiRenderBounds));

    // Kaiser
    kaiserAlpha = 2.0f;//13.044230f; // 1.4151f;
    accuracy = 1E-6;
    piAlpha = M_PI * kaiserAlpha;
    
    // Prolate
    prolateC = 3 * M_PI;
    prolateAlpha = 1;
    numTerms = 16; // default: 16
}

void initGridder(void) {
   
    kernalBuffer = (GLfloat*) malloc(sizeof (GLfloat) * kernelSize * kernelSize);
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
    
    createKernel();
    
    //kernal TEXTURE
    kernalTextureID = idArray[1];
    glBindTexture(GL_TEXTURE_2D, kernalTextureID);
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glEnable(GL_TEXTURE_2D);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_R32F, (int) kernelSize, (int) kernelSize, 0, GL_RED, GL_FLOAT, kernalBuffer);
    glBindTexture(GL_TEXTURE_2D, 0);

    
//    glTexEnvi(GL_POINT_SPRITE, GL_COORD_REPLACE, GL_TRUE);
//    glPointSize((int) kernelSize);
//    glEnable(GL_POINT_SPRITE);
    
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
        
        int randomKernel = (int)((float)rand()/RAND_MAX * 121)+7;
        
        while(randomKernel % 2 == 0)
        {
            randomKernel = (int)((float)rand()/RAND_MAX * 121)+7;
        }
        
        visibilities[i] = (float) (rand() % (int) gridDimension);
        visibilities[i + 1] = (float) (rand() % (int) gridDimension);
        visibilities[i + 2] = (float) 75.0f;//randomKernel;
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

    glBindTexture(GL_TEXTURE_2D, kernalTextureID);
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
    glBindTexture(GL_TEXTURE_2D, 0);
    glUseProgram(0);
    glDisable(GL_BLEND);
    
    iterationCount++;
    
    if(iterationCount == dumpTime)
    {
        glFinish();
        glReadPixels(0, 0, gridDimension, gridDimension, GL_RGBA, GL_FLOAT, gridBuffer);
//        printf("Pixel1: %f %f %f %f\n", gridBuffer[0], gridBuffer[1], gridBuffer[2], gridBuffer[3]);
//        printf("Pixel2: %f %f %f %f\n", gridBuffer[4], gridBuffer[5], gridBuffer[6], gridBuffer[7]);
        
//        for(int i = 0; i < kernelSize * 3; i+=3)
//            printf("Pixel%d: %f %f %f\n", i, gridBuffer[i], gridBuffer[i+1], gridBuffer[i+2]);
    
        iterationCount = 0;
        
//        glClearColor(0.0f, 0.0f, 0.0f, 0.0f);
//        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    }
    
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
    //glPopAttrib();

    //DRAW RENDERING
     glViewport(0, 0, gridDimension, gridDimension);
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

void createKernel(void)
{    
    if(kernelType == KAISER || kernelType == PROLATE)
    {
        float half_width_adam = ((kernelSize - 1.0f)) / 2.0f;

        for (int row = 0; row < kernelSize; row++) {
            
            float rowKernelWeight = calculateKernelWeight(row/half_width_adam-1.0f);

            printf("row: %d %f %f\n",row, row/half_width_adam-1.0f, rowKernelWeight);
//            printf("%f ", rowKernelWeight);
//            printf("\n\n");
            
            for (int col = 0, c = 0; col < kernelSize; col++, c++) {
                float colWeight = c/half_width_adam-1.0f;
                kernalBuffer[(int) kernelSize * row + col] = rowKernelWeight * calculateKernelWeight(colWeight);
//                kernalBuffer[(int) kernelSize * 4 * row + (col + 1)] = 1.0f;
//                kernalBuffer[(int) kernelSize * 4 * row + (col + 2)] = 0.0f;
//                kernalBuffer[(int) kernelSize * 4 * row + (col + 3)] = 0.0f;
                
//               printf("%f ", kernalBuffer[(int) kernelSize * 4 * row + col]);
            }
//            printf("\n\n");
        }
//        printf("\n\n\n");
    }
    else
    {
        for (int i = 0; i < kernelSize * kernelSize; i++) {
            kernalBuffer[i] = (GLfloat) rand() / RAND_MAX;
//            kernalBuffer[i + 1] = (GLfloat) rand() / RAND_MAX;
//            kernalBuffer[i + 2] = (GLfloat) rand() / RAND_MAX;
//            kernalBuffer[i + 3] = 1.0f;
        }
    }
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
    setenv("DISPLAY", ":1", 11.0);
    glutInit(&argc, argv);
    glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE | GLUT_ALPHA);
    glutInitWindowSize(gridDimension, gridDimension);
    glutInitWindowPosition(0, 0);
    glutCreateWindow("Gridder");
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
    if(counter == 10)
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