
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

#define FRAGMENT_SHADER " \
  #version 430\n \
  precision highp float; \
  uniform sampler2D kernalTex;\
  in vec2 fComplex; \
  void main() { \
    float kernelLookup = texture(kernalTex,gl_PointCoord).r; \
    gl_FragColor.r = kernelLookup; \
    gl_FragColor.gb = kernelLookup * fComplex; \
    gl_FragColor.a = 1.0; \
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

