
/* 
 * File:   gpu.h
 * Author: adam
 *
 * Created on 1 August 2017, 11:23 AM
 */

#ifndef GPU_H
#define GPU_H

/*
 * Inclusion of 0.04 to gl_PointSize is to accommodate overflow
 * of kernel texture weights when smearing a fixed sized kernel texture. 
 */
#define VERTEX_SHADER " \
  #version 430\n \
  precision highp float; \
  in vec3 position; \
  in vec2 complex; \
  out vec2 fComplex; \
  void main() { \
     gl_Position.rg = (2.0*position.rg + 1.0) / %f -1.0; \
     gl_Position.ba = vec2(0.5,1.0);\
     gl_PointSize = position.b + 0.04; \
     fComplex = complex; \
  }"

// (a+bi)(c+di)=(ac-bd)+(bc+ad)i

// R: Kernel Weight (real)
// G: Kernel Weight (real) * Visibility Real
// B: Kernel Weight (imaginary) * Visibility Imaginary
// A: Kernel Weight (imaginary) accumulation
#define FRAGMENT_SHADER " \
  #version 430\n \
  precision highp float; \
  uniform sampler2D kernalTex;\
  in vec2 fComplex; \
  void main() { \
    vec2 kernelLookup = texture(kernalTex,gl_PointCoord).rg; \
    gl_FragColor.r = kernelLookup.r; \
    gl_FragColor.gb = vec2(kernelLookup.r * fComplex.r - kernelLookup.g * fComplex.g, \
                                 kernelLookup.g * fComplex.r + kernelLookup.r * fComplex.g); \
    gl_FragColor.a = kernelLookup.g; \
  }"

#define FRAGMENT_SHADER_RENDER " \
  #version 430\n \
  uniform sampler2D destTex; \
  in vec2 texCoord;\
  void main() { \
    gl_FragColor = texture(destTex, texCoord); \
  }"

#define VERTEX_SHADER_RENDER " \
  #version 430\n \
  in vec4 position;\
  out vec2 texCoord;\
  void main() { \
     gl_Position = position; \
    gl_Position.rg = (2.0*position.rg + 1.0) / %f -1.0; \
    texCoord = gl_Position.rg*0.5 + 0.5;\
  }"

#endif /* GPU_H */

