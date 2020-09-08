
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
  out float pointsize;\
  void main() { \
     gl_Position.rg = ((position.rg*uvScale) / gridCenter) + vec2(gridCenterOffset,gridCenterOffset); \
     wPlane = sqrt(abs(position.b*wScale)) * wStep + (0.5 * wStep);\
     float w_half_support = abs(wToMaxSupportRatio * position.b) + minSupportOffset; \
     conjugate = -sign(position.b);\
     gl_PointSize = (w_half_support * 2.0) + 1.0;\
     fComplex = complex.rg * complex.b; \
     pointsize = gl_PointSize;\
  }"

//NEED TO MAKE UNIFORM VALUE TO PASS IN HALF TEX SIZE TO CHANGE 30
//vec2 coord = (1.0 + 2.0*(2.0 * half_tex_size - 1.0)*abs(gl_PointCoord.xy-0.5))/(2.0*half_tex_size); 
#define FRAGMENT_SHADER_REFLECT_VEC2 " \
  #version 430\n \
  precision highp float; \
  uniform sampler3D kernelTex;\
  in vec2 fComplex; \
  in float wPlane; \
  in float conjugate; \
  in float pointsize;\
  void main() { \
    float half_tex_size = 64.0; \
    vec2 coord = 1.0/(2.0*half_tex_size) + 2.0*pointsize*abs(gl_PointCoord.xy-0.5)/(pointsize+1.0);\
    vec2 kernelLookup = texture(kernelTex,vec3(coord.xy,wPlane)).rg; \
    kernelLookup.g = kernelLookup.g * conjugate;\
    gl_FragColor.rg = vec2(kernelLookup.r * fComplex.r - kernelLookup.g * fComplex.g, \
                                kernelLookup.g * fComplex.r + kernelLookup.r * fComplex.g); \
    gl_FragColor.g = coord.x; \
  }"

#define FRAGMENT_SHADER_REFLECT_VEC4 " \
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


