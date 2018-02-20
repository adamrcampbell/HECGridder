
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

#include "gridder.h"
#include "gpu.h"

#ifndef M_PI
    #define M_PI 3.14159265358979323846264338327
#endif

// NAG Dataset configurations
//
//  File:           el82-70.txt
//  Min Support:    4.0
//  Max Support:    44.0
//  Max W:          6746.082031 //7000
//  Max Plane:      339.0
//  Cell Size Rad:  0.000006
//  Grid Size       18000
//
//  File:           el56-82.txt
//  Min Support:    4
//  Max Support:    72
//  Max W:          11937.875977   //12000
//  Max Plane:      601
//  Cell Size Rad:  0.000006
//
//  File:           el30-56
//  Min Support:    4
//  Max Support:    95
//  Max W:          -18309 //19000
//  Max Plane:      922
//  Cell Size Rad:  0.000006
//
//  File:           synthetic.txt
//  Grid Size:      100000
//  Min Support:    4
//  Max Support:    36
//  Max W:          38971
//  Max Plane:      714
//  Cell Size Rad:  0.000004

// ADAMS TO DO LIST:
// - W projection doesnt seem to hit max W value

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

// Global gridder configuration
Config config;

void initConfig(void) 
{
    // Global
    windowDisplay = 900;
    config.kernelTexSize = 128;                             // kernelTexSize >= kernelMaxFullSupport
    config.kernelResolutionSize = 256;                      // Always a power of 2 greater than the textureSize MINIMUM
    config.gridDimension = 18000;
    config.kernelMaxFullSupport = (44.0f * 2.0f) + 1.0f;    // kernelMaxFullSupport <= kernelResolutionSize
    config.kernelMinFullSupport = (4.0f * 2.0f) + 1.0f;
    config.visibilityCount = 1;
    // config.visibilityCount = 1;//31395840;
    config.numVisibilityParams = 5;
    config.visibilitiesFromFile = true;
    config.displayDumpTime = 50;
    config.visibilitySourceFile = "datasets/el82-70.txt"; //"el82-70_vis.txt";
    // Gui
    config.refreshDelay = 0;
    float gridDimFloat = (float) config.gridDimension;
    GLfloat renderTemp[8] = {
        -gridDimFloat/2.0f, -gridDimFloat/2.0f,
        -gridDimFloat/2.0f, gridDimFloat/2.0f,
        gridDimFloat/2.0f, -gridDimFloat/2.0f,
        gridDimFloat/2.0f, gridDimFloat/2.0f
    };
    memcpy(guiRenderBounds, renderTemp, sizeof (guiRenderBounds));
    
    // W-Projection
    config.wProjectionMaxW = 7000.0f;
    config.cellSizeRad = 0.000006;
    config.wProjectNumPlanes = 339;//(int) (config.wProjectionMaxW * fabs(sin(config.cellSizeRad * (double) config.gridDimension / 2.0)));
    config.wScale = pow(config.wProjectNumPlanes-1, 2.0) / config.wProjectionMaxW; // (config.wProjectNumPlanes * config.wProjectNumPlanes) / config.wProjectionMaxW;
    config.wProjectionStep = config.wProjectionMaxW / config.wScale;
    config.fieldOfView = config.cellSizeRad *  config.gridDimension;
    config.uvScale = config.fieldOfView * 1.0; // second factor for scaling
    config.wToMaxSupportRatio = (config.kernelMaxFullSupport - config.kernelMinFullSupport) / config.wProjectionMaxW;
    
    printf("W Scale: %f\n", config.wScale);
    printf("W Max Support Ratio: %f\n", config.wToMaxSupportRatio);
}

void initGridder(void) 
{    
    glClampColorARB(GL_CLAMP_VERTEX_COLOR_ARB, GL_FALSE);
    glClampColorARB(GL_CLAMP_READ_COLOR_ARB, GL_FALSE);
    glClampColorARB(GL_CLAMP_FRAGMENT_COLOR_ARB, GL_FALSE);

    srand(time(NULL));
    int size = 4 * config.gridDimension*config.gridDimension;
    gridBuffer = (GLfloat*) malloc(sizeof (GLfloat) * size);

    for (int i = 0; i < size; i += 4) 
    {
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
    
    kernelBuffer = calloc(config.kernelTexSize * config.kernelTexSize * config.wProjectNumPlanes, sizeof(FloatComplex));
    createWProjectionPlanes(kernelBuffer);
    
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
    glTexImage3D(GL_TEXTURE_3D, 0,  GL_RG32F, config.kernelTexSize, config.kernelTexSize, (int) config.wProjectNumPlanes, 0, GL_RG, GL_FLOAT, kernelBuffer);
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


void setShaderUniforms(void)
{
    printf("SETTING THE SHADER UNIFORMS\n");
    glUseProgram(sProgram);
    glUniform1f(uMinSupportOffset, config.kernelMinFullSupport);
    glUniform1f(uWToMaxSupportRatio, (config.kernelMaxFullSupport-config.kernelMinFullSupport)/config.wProjectionMaxW); //(maxSuppor-minSupport) / maxW
    glUniform1f(uGridSize, config.gridDimension);
    glUniform1f(uWScale, config.wScale);
    glUniform1f(uUVScale, config.uvScale);
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
            visibilities[i + 2] = 7000.0f;//(float) randomKernel;
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

DoubleComplex complexAdd(DoubleComplex x, DoubleComplex y)
{
    DoubleComplex z;
    z.real = x.real + y.real;
    z.imaginary = x.imaginary + y.imaginary;
    return z;
}

DoubleComplex complexSubtract(DoubleComplex x, DoubleComplex y)
{
    DoubleComplex z;
    z.real = x.real - y.real;
    z.imaginary = x.imaginary - y.imaginary;
    return z;
}

DoubleComplex complexMultiply(DoubleComplex x, DoubleComplex y)
{
    DoubleComplex z;
    z.real = x.real*y.real - x.imaginary*y.imaginary;
    z.imaginary = x.imaginary*y.real + x.real*y.imaginary;
    return z;
}

DoubleComplex complexConjugateExp(double ph)
{
    return (DoubleComplex) {.real = cos((double)(2.0*M_PI*ph)), .imaginary = -sin((double)(2.0*M_PI*ph))};
}

double complexMagnitude(DoubleComplex x)
{
    return sqrt(x.real * x.real + x.imaginary * x.imaginary);
}

int calcWFullSupport(double w, double wToMaxSupportRatio, double minSupport)
{
    // Calculates the full support width of a kernel for w term
    return (int) (fabs(wToMaxSupportRatio * w) + minSupport);
}

void normalizeKernel(DoubleComplex *kernel, int resolution, int support)
{
    // Get sum of magnitudes
    double magnitudeSum;
    int r, c;
    for(r = 0; r < resolution; r++)
        for(c = 0; c < resolution; c++)
            magnitudeSum += complexMagnitude(kernel[r * resolution + c]);
    
    // Normalize weights
    for(r = 0; r < resolution; r++)
        for(c = 0; c < resolution; c++)
            kernel[r * resolution + c] = normalizeWeight(kernel[r * resolution + c], magnitudeSum, resolution, support);
}

DoubleComplex normalizeWeight(DoubleComplex weight, double mag, int resolution, int support)
{
    DoubleComplex normalized;
    double t2 = (resolution*resolution)/(support*support);
    normalized.real = (weight.real / mag) * t2;
    normalized.imaginary = (weight.imaginary / mag) * t2;
    return normalized;
}

void interpolateKernel(DoubleComplex *source, DoubleComplex* dest, int origSupport, int texSupport)
{   
    // Perform bicubic interpolation
    InterpolationPoint neighbours[16];
    InterpolationPoint interpolated[4];
    float xShift, yShift;
    
    for(int y = 0; y < texSupport; y++)
    {
        yShift = calcInterpolateShift(y, texSupport, -0.5)+(getShift(texSupport)/(4.0));
        
        for(int x = 0; x < texSupport; x++)
        {
            xShift = calcInterpolateShift(x, texSupport, -0.5)+(getShift(texSupport)/(4.0));
            getBicubicNeighbours(x, y, neighbours, origSupport, texSupport, source);
            
            for(int i  = 0; i < 4; i++)
            {
                InterpolationPoint newPoint = (InterpolationPoint) {.xShift = xShift, .yShift = neighbours[i*4].yShift};
                newPoint = interpolateCubicWeight(neighbours, newPoint, i*4, origSupport, true);
                interpolated[i] = newPoint;
                
//                printf("[%f, %f] ", newPoint.yShift, newPoint.xShift);
//                printf("%f %f %f %f %f\n", neighbours[0].xShift, neighbours[1].xShift, newPoint.xShift, neighbours[2].xShift, neighbours[3].xShift);
//                printf("%f %f %f %f %f\n", neighbours[0].yShift, neighbours[1].yShift, newPoint.yShift, neighbours[2].yShift, neighbours[3].yShift);
            }
//            printf("\n");

            // printf("[X: %f, Y: %f] ", xShift, yShift);
            
            InterpolationPoint final = (InterpolationPoint) {.xShift = xShift, .yShift = yShift};
            final = interpolateCubicWeight(interpolated, final, 0, origSupport, false);
            int index = y * texSupport + x;
            dest[index] = (DoubleComplex) {.real = final.weight.real, .imaginary = final.weight.imaginary};
        }
//        printf("\n");
    }
}

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
    
    // numWPlanes = 2;//numWPlanes-1;
    int plane = numWPlanes-1;
    printf("Num W Planes: %d\n", numWPlanes);
    for(int iw = 0; iw < numWPlanes; iw++)
    {        
        // Calculate w term and w specific support size
        double w = iw * iw / wScale;
        double fresnel = w * ((0.5 * config.fieldOfView)*(0.5 * config.fieldOfView));
        printf("CreateWTermLike: For w = %f, field of view = %f, fresnel number = %f\n", w, config.fieldOfView, fresnel);
        int wFullSupport = calcWFullSupport(w, config.wToMaxSupportRatio, config.kernelMinFullSupport);
        // Calculate Prolate Spheroidal
        createScaledSpheroidal(spheroidal, wFullSupport, convHalf);
        
//        // Prints prolate spheroidal
//        for(int i = 0; i < convolutionSize; i++)
//        {
//            for(int j = 0; j < convolutionSize; j++)
//            {
//                printf("%f ", spheroidal[i] * spheroidal[j]);
//            }
//            printf("\n");
//        }
//        
//        for(int i = 0; i < convolutionSize; i++)
//            printf("%f\n", spheroidal[i]);
        
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
       
        // Normalize the kernel
        normalizeKernel(shift, convolutionSize, wFullSupport);
        
        if(iw == plane)
            saveKernelToFile("output/wproj_%f_normalized_%d.csv", w, convolutionSize, shift);
        
        DoubleComplex *interpolated = calloc(textureSupport * textureSupport, sizeof(DoubleComplex));
        interpolateKernel(shift, interpolated, convolutionSize, textureSupport);
        
        if(iw == plane)
            saveKernelToFile("output/wproj_%f_interpolated_%d.csv", w, textureSupport, interpolated);
        
        // Bind interpolated kernel to texture matrix
        for(int y = 0; y < textureSupport; y++)
        {
            for(int x = 0; x < textureSupport; x++)
            {
                DoubleComplex interpWeight = interpolated[y * textureSupport + x];
                FloatComplex weight = (FloatComplex) {.real = (float) interpWeight.real, .imaginary = (float) interpWeight.imaginary};
                int index = (iw * textureSupport * textureSupport) + (y * textureSupport) + x;
                wTextures[index] = weight;
            }
        }
        
        free(interpolated);
        memset(screen, 0, convolutionSize * convolutionSize * sizeof(DoubleComplex));
        memset(shift, 0, convolutionSize * convolutionSize * sizeof(DoubleComplex));
    }
    
    free(spheroidal);
    free(screen);
    free(shift);
}

void createScaledSpheroidal(double *spheroidal, int wFullSupport, int convHalf)
{
    int wHalfSupport = wFullSupport/2;
    double *nu = calloc(wFullSupport, sizeof(double));
    double *tempSpheroidal = calloc(wFullSupport, sizeof(double));
    // Calculate steps
    for(int i = 0; i < wFullSupport; i++)
    {
        nu[i] = fabs(calcSpheroidalShift(i, wFullSupport));
//        printf("%f\n", nu[i]);
    }
//    printf("\n\n");
        
    // Calculate curve from steps
    calcSpheroidalCurve(nu, tempSpheroidal, wFullSupport);
    
    // Zero out first weight
    tempSpheroidal[0] = 0.0;
    // Zero out last weight to balance spheroidal
    if(wFullSupport % 2 != 0)
        tempSpheroidal[wFullSupport-1] = 0.0;
    
    // Bind weights to middle
    for(int i = convHalf-wHalfSupport; i <= convHalf+wHalfSupport; i++)
        spheroidal[i] = tempSpheroidal[i-(convHalf-wHalfSupport)];    
    
    free(tempSpheroidal);
    free(nu);
}

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
            
            // Note: what happens when W = 0?
                
            screen[index].real *= taper;
            screen[index].imaginary *= taper;
        }
    }
}

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
    
    // Refactor for complex multiplication and subtraction
    DoubleComplex t0 = complexMultiply(complexMultiply(complexMultiply(w0, complexSubtract(interpShift, p1)), complexSubtract(interpShift, p2)), 
            complexSubtract(interpShift, p3));
    DoubleComplex t1 = complexMultiply(complexMultiply(complexMultiply(w1, complexSubtract(interpShift, p0)), complexSubtract(interpShift, p2)), 
            complexSubtract(interpShift, p3));
    DoubleComplex t2 = complexMultiply(complexMultiply(complexMultiply(w2, complexSubtract(interpShift, p0)), complexSubtract(interpShift, p1)),
            complexSubtract(interpShift, p3));
    DoubleComplex t3 = complexMultiply(complexMultiply(complexMultiply(w3, complexSubtract(interpShift, p0)), complexSubtract(interpShift, p1)), 
            complexSubtract(interpShift, p2));
    // Refactor for complex addition
    newPoint.weight = complexAdd(complexAdd(complexAdd(t0, t1), t2), t3);
    return newPoint;
}

void getBicubicNeighbours(int x, int y, InterpolationPoint *neighbours, int origFullSupport, int interpFullSupport, DoubleComplex* matrix)
{
    // Transform x, y into scaled shift
    float shiftX = calcInterpolateShift(x+1, interpFullSupport, -0.5);
    float shiftY = calcInterpolateShift(y+1, interpFullSupport, -0.5);
//    printf("Populating element at Row: %d and Col: %d\n", y, x);
//    printf("Row Shift: %f, Col Shift: %f\n", shiftY, shiftX);
    // Get x, y from scaled shift 
    int scaledPosX = calcPosition(shiftX, origFullSupport)-1;
    int scaledPosY = calcPosition(shiftY, origFullSupport)-1;
//     printf("X: %d, Y: %d\n", scaledPosX, scaledPosY);
    // Get 16 nInterpolationPointeighbours
    for(int r = scaledPosY - 1, i = 0; r < scaledPosY + 3; r++)
    {
        for(int c = scaledPosX - 1; c < scaledPosX + 3; c++)
        {
            InterpolationPoint n = (InterpolationPoint) {.xShift = calcShift(c, origFullSupport, -1.0), .yShift = calcShift(r, origFullSupport, -1.0)};
            
            if(c < 0 || c > origFullSupport || r < 0 || r > origFullSupport)
                n.weight = (DoubleComplex) {.real = 0.0, .imaginary = 0.0};
            else
                n.weight = matrix[r * origFullSupport + c];
            
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

float calcInterpolateShift(int index, int width, float start)
{
    return start + ((float)index/(float)width);
}

float calcShift(int index, int width, float start)
{
    return start + index * getShift(width);
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
    return (int) floor(((x+1.0f)/2.0f) * (scalerWidth));
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
