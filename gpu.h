
/*
 * 
 * Authors: Dr. Seth Hall, Dr. Andrew Ensor, Adam Campbell
 * Auckland University of Technology - AUT
 * High Performance Laboratory
 * 
 */

#ifndef GPU_H
#define GPU_H

/*
 * Shader Program: VERTEX_SHADER (invoked once per bound visibility)
 * --------------------
 * For each visibility bound to GPU, this shader calculates several parameters:
 * >>> visibility UV coordinates are transformed to grid coordinates (position.rg)
 * >>> visibility W coordinate is used for calculating which W convolution kernel
 *        should be applied (position.b) (negative W coordinate is handled by conjugate variable)
 * >>> visibility full support is determined by gl_PointSize (odd number to ensure peak)
 *        note: addition of 0.04 is used to accommodate overflow of convolved visibilities
 *        on centered and shifted pixels. This will calculate how many fragments will be required
 *        for convolving of visibility to grid, per visibility.
 * >>> visibility real and imaginary is bound to fComplex (from complex.rg),
 *        and is multiplied by the weight (complex.b)
 */
#define VERTEX_SHADER " \
  #version 430\n \
  precision highp float; \
  uniform float minSupportOffset;\
  uniform float wToMaxSupportRatio;\
  uniform float gridCenter;\
  uniform float gridCenterOffset; \
  uniform float wScale;\
  uniform float wStep; \
  uniform float uvScale;\
  uniform float numPlanes;\
  in vec3 position; \
  in vec3 complex; \
  out vec2 fComplex; \
  out float wPlane;\
  out float conjugate;\
  void main() { \
     gl_Position.rg = ((position.rg*uvScale) / gridCenter) + vec2(gridCenterOffset,gridCenterOffset); \
     wPlane = sqrt(abs(position.b*wScale)) * wStep + (0.5 * wStep);\
     float wSupport = abs(wToMaxSupportRatio*position.b) + minSupportOffset; \
     conjugate = -sign(position.b);\
     gl_PointSize = wSupport + (1.0 - mod(wSupport, 2.0));\
     fComplex = complex.rg * complex.b; \
  }"
///gl_Position.rg = ((position.rg*uvScale) / gridCenter) + gridCenterOffset; 
//wSupport + (1.0 - mod(wSupport, 2.0))
//wPlane = sqrt(abs(position.b*wScale)) * wStep; 
/*
 * Shader Program: VERTEX_SHADER_SNAP (invoked once per bound visibility)
 * --------------------
 * Same as above (VERTEX_SHADER), however this version uses the round function
 * when calculating grid position for visibility and specific w plane required.
 */
#define VERTEX_SHADER_SNAP " \
  #version 430\n \
  precision highp float; \
  uniform float minSupportOffset;\
  uniform float wToMaxSupportRatio;\
  uniform float gridCenter;\
  uniform float gridCenterOffset; \
  uniform float wScale;\
  uniform float wStep; \
  uniform float uvScale;\
  uniform float numPlanes;\
  in vec3 position; \
  in vec3 complex; \
  out vec2 fComplex; \
  out float wPlane;\
  out float conjugate;\
  void main() { \
     gl_Position.rg = (round(position.rg*uvScale) / gridCenter) + vec2(gridCenterOffset, -gridCenterOffset); \
     wPlane = round(sqrt(abs(position.b*wScale))) * wStep; \
     float wSupport = abs(wToMaxSupportRatio*position.b) + minSupportOffset; \
     conjugate = -sign(position.b);\
     gl_PointSize = wSupport + (1.0 - mod(wSupport, 2.0)); \
     fComplex = complex.rg * complex.b; \
  }"

// R: Texture real
// G: Texture real multiplied with visibility real
// B: Texture imaginary multiplied with visibility imaginary
// A: Texture imaginary

/*
 * Shader Program: VERTEX_SHADER_SNAP (invoked for each fragment (kernel element)
 * in two dimensional convolution kernel, per visibility) - (gl_PointSize * gl_PointSize)
 * --------------------
 * All convolution kernels are stored in a single 3d texture (sprite). Input wPlane parameter
 * determines the z-index of the desired w projection kernel, gl_PointCoord.xy determines the row
 * and column element within the convolution kernel to be used. This shader assumes that the bound
 * OpenGL frame buffer is set to accumulation mode.
 * 
 * Each fragment outputs an RGBA vector, consisting of the following:
 * >>> R: Texture real weight (kernel weight)
 * >>> G: Visibility real complex multiplication with looked up kernel element
 * >>> B: Visibility imag complex multiplication with looked up kernel element
 * >>> A: Texture imaginary weight (kernel weight)
 */
#define FRAGMENT_SHADER_RADIAL " \
  #version 430\n \
  precision highp float; \
  uniform sampler2D kernelTex;\
  in vec2 fComplex; \
  in float wPlane; \
  in float conjugate; \
  void main() { \
    vec2 coord = 2.0*gl_PointCoord.xy - 1.0;\
    coord *= coord;\
    float radialLookup = clamp(sqrt(coord.x+coord.y),0.0,1.0); \
    vec2 kernelLookup = texture(kernelTex,vec2(radialLookup,wPlane)).rg; \
    kernelLookup.g = kernelLookup.g * conjugate;\
    gl_FragColor.ra = kernelLookup.rg; \
    gl_FragColor.gb = vec2(kernelLookup.r * fComplex.r - kernelLookup.g * fComplex.g, \
                                kernelLookup.g * fComplex.r + kernelLookup.r * fComplex.g); \
  }"

#define FRAGMENT_SHADER_REFLECT " \
  #version 430\n \
  precision highp float; \
  uniform sampler3D kernelTex;\
  in vec2 fComplex; \
  in float wPlane; \
  in float conjugate; \
  void main() { \
    vec2 coord = abs(2.0*gl_PointCoord.xy - 1.0);\
    vec2 kernelLookup = texture(kernelTex,vec3(coord.xy,wPlane)).rg; \
    kernelLookup.g = kernelLookup.g * conjugate;\
    gl_FragColor.ra = kernelLookup.rg; \
    gl_FragColor.gb = vec2(kernelLookup.r * fComplex.r - kernelLookup.g * fComplex.g, \
                                kernelLookup.g * fComplex.r + kernelLookup.r * fComplex.g); \
  }"

#define FRAGMENT_SHADER " \
  #version 430\n \
  precision highp float; \
  uniform sampler3D kernelTex;\
  in vec2 fComplex; \
  in float wPlane; \
  in float conjugate; \
  void main() { \
    vec2 kernelLookup = texture(kernelTex,vec3(gl_PointCoord.xy,wPlane)).rg; \
    kernelLookup.g = kernelLookup.g * conjugate;\
    gl_FragColor.ra = kernelLookup.rg; \
    gl_FragColor.gb = vec2(kernelLookup.r * fComplex.r - kernelLookup.g * fComplex.g, \
                                 kernelLookup.g * fComplex.r + kernelLookup.r * fComplex.g); \
  }"


//    gl_FragColor.ra = vec2(kernelLookup.r * fComplex.r - kernelLookup.g * fComplex.g, \
//                                 kernelLookup.g * fComplex.r + kernelLookup.r * fComplex.g); \
// Performs rendering to screen (not gridding)
#define FRAGMENT_SHADER_RENDER " \
  #version 430\n \
  uniform sampler2D destTex; \
  in vec2 texCoord;\
  void main() { \
    gl_FragColor = texture(destTex, texCoord); \
  }"

// Performs rendering to screen (not gridding)
#define VERTEX_SHADER_RENDER " \
  #version 430\n \
  in vec2 position;\
  out vec2 texCoord;\
  uniform float gridSize;\
  void main() { \
    gl_Position.xy = position.xy;\
    texCoord = position.xy*0.5 + 0.5;\
  }"

#endif /* GPU_H */


