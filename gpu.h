

#ifndef GPU_H
#define GPU_H

/*
 * Inclusion of 0.04 to gl_PointSize is to accommodate overflow
 * of kernel texture weights when smearing a fixed sized kernel texture.
 */

#define VERTEX_SHADER " \
  #version 430\n \
  precision highp float; \
  uniform float minSupportOffset;\
  uniform float wToMaxSupportRatio;\
  uniform float gridSize;\
  uniform float wScale;\
  uniform float uvScale;\
  in vec3 position; \
  in vec2 complex; \
  out vec2 fComplex; \
  out float wPlane;\
  void main() { \
     gl_Position.rg = (position.rg*uvScale) / ((gridSize-0.5)/2.0); \
     wPlane = sqrt(abs(position.b)*wScale) * sign(position.b); \
     gl_PointSize = abs(wToMaxSupportRatio*position.b) + minSupportOffset + 0.04; \
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
  uniform sampler3D kernalTex;\
  in vec2 fComplex; \
  in float wPlane; \
  void main() { \
    vec2 kernelLookup = texture(kernalTex,vec3(gl_PointCoord.xy,abs(wPlane))).rg; \
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
  uniform float gridSize;\
  void main() { \
    gl_Position.rg = (position.rg) / ((gridSize-0.5)/2.0); \
    texCoord = gl_Position.rg*0.5 + 0.5;\
  }"

#endif /* GPU_H */