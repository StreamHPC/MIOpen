/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2017-2025 Advanced Micro Devices, Inc.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 *******************************************************************************/

// Disable specific warnings
#ifdef __clang__
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wconditional-uninitialized"
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wsometimes-uninitialized"
#endif

#include "batchnorm_functions.h"

__attribute__((reqd_work_group_size(MIO_BN_GRP0, MIO_BN_GRP1, MIO_BN_GRP2))) __kernel void
MIOpenBatchNormFwdInferSpatialEst(const __global _FLOAT* __restrict in, /* x input */
                                  __global _FLOAT* __restrict out,      /* y output */
                                  const __global _FLOAT_PREC* __restrict estimatedMean,
                                  const __global _FLOAT_PREC* __restrict estimatedVariance,
                                  const __global _FLOAT_PREC* __restrict scale,
                                  const __global _FLOAT_PREC* __restrict bias,
                                  double epsilon,
                                  unsigned int c,
                                  unsigned int hw,
                                  unsigned int batchSize,
                                  unsigned int cStride,
                                  unsigned int hwStride,
                                  unsigned int batchStride)
{
    unsigned int xgid = get_global_id(0);
    unsigned int ygid = get_global_id(1);

#if MIO_LAYOUT_NHWC
    (void)cStride;

    if(xgid * 4 >= c || ygid >= hw)
        return;

    unsigned int index;
    _FLOAT_PREC4 mean, variance, invVariance;
    _FLOAT_PREC4 pscale, pbias;
    _FLOAT_PREC4 inhat;
    _FLOAT4 value;

    mean        = *((const __global _FLOAT_PREC4*)(estimatedMean + xgid * 4));
    variance    = *((const __global _FLOAT_PREC4*)(estimatedVariance + xgid * 4));
    pscale      = *((const __global _FLOAT_PREC4*)(scale + xgid * 4));
    pbias       = *((const __global _FLOAT_PREC4*)(bias + xgid * 4));
    invVariance = rsqrt(fabs(variance + (_FLOAT_PREC4)epsilon));

    for(int n = 0; n < batchSize; n++)
    {
        index = (n * batchStride) + (xgid * 4) + (ygid * hwStride);
        value = *((const __global _FLOAT4*)(in + index));

        inhat = (_FLOAT_PREC4)(FLOAT2FLOATPREC(value.x),
                               FLOAT2FLOATPREC(value.y),
                               FLOAT2FLOATPREC(value.z),
                               FLOAT2FLOATPREC(value.w));
        inhat = (inhat - mean) * invVariance;
        inhat = mad(pscale, inhat, pbias);
        value = (_FLOAT4)(FLOATPREC2FLOAT(inhat.x),
                          FLOATPREC2FLOAT(inhat.y),
                          FLOATPREC2FLOAT(inhat.z),
                          FLOATPREC2FLOAT(inhat.w));

        *((__global _FLOAT4*)(out + index)) = value;
    }
#else
    (void)hwStride;

    if(xgid >= c || ygid * 4 >= hw)
        return;

    unsigned int index;
    _FLOAT_PREC mean, variance, invVariance;
    _FLOAT_PREC pscale, pbias;
    _FLOAT_PREC4 inhat;
    _FLOAT4 value;

    mean        = *((const __global _FLOAT_PREC*)(estimatedMean + xgid));
    variance    = *((const __global _FLOAT_PREC*)(estimatedVariance + xgid));
    pscale      = *((const __global _FLOAT_PREC*)(scale + xgid));
    pbias       = *((const __global _FLOAT_PREC*)(bias + xgid));
    invVariance = rsqrt(fabs(variance + (_FLOAT_PREC)epsilon));

    for(int n = 0; n < batchSize; n++)
    {
        index = (n * batchStride) + (xgid * cStride) + (ygid * 4);
        value = *((const __global _FLOAT4*)(in + index));

        inhat = (_FLOAT_PREC4)(FLOAT2FLOATPREC(value.x),
                               FLOAT2FLOATPREC(value.y),
                               FLOAT2FLOATPREC(value.z),
                               FLOAT2FLOATPREC(value.w));
        inhat = (inhat - mean) * invVariance;
        inhat = mad(pscale, inhat, pbias);
        value = (_FLOAT4)(FLOATPREC2FLOAT(inhat.x),
                          FLOATPREC2FLOAT(inhat.y),
                          FLOATPREC2FLOAT(inhat.z),
                          FLOATPREC2FLOAT(inhat.w));

        *((__global _FLOAT4*)(out + index)) = value;
    }
#endif

} // end spatial norm

#ifdef __clang__
#pragma clang diagnostic pop
#pragma clang diagnostic pop
#endif
