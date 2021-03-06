/******************************************************************************
* Copyright (c) Intel Corporation - All rights reserved.                      *
* This file is part of the LIBXSMM library.                                   *
*                                                                             *
* For information on the license, see the LICENSE file.                       *
* Further information: https://github.com/hfp/libxsmm/                        *
* SPDX-License-Identifier: BSD-3-Clause                                       *
******************************************************************************/
/* Hans Pabst (Intel Corp.)
******************************************************************************/
#ifndef LIBXSMM_CPUID_H
#define LIBXSMM_CPUID_H

#include "libxsmm_macros.h"

/**
 * Enumerates the available target architectures and instruction
 * set extensions as returned by libxsmm_get_target_archid().
 */
#define LIBXSMM_TARGET_ARCH_UNKNOWN 0
#define LIBXSMM_TARGET_ARCH_GENERIC 1
#define LIBXSMM_X86_GENERIC      1002
#define LIBXSMM_X86_SSE3         1003
#define LIBXSMM_X86_SSE4         1004
#define LIBXSMM_X86_AVX          1005
#define LIBXSMM_X86_AVX2         1006
#define LIBXSMM_X86_AVX512       1007
#define LIBXSMM_X86_AVX512_MIC   1010 /* KNL */
#define LIBXSMM_X86_AVX512_KNM   1011
#define LIBXSMM_X86_AVX512_CORE  1020 /* SKX */
#define LIBXSMM_X86_AVX512_CLX   1021
#define LIBXSMM_X86_AVX512_CPX   1022
#define LIBXSMM_X86_ALLFEAT      1999 /* all features supported which are used anywhere in LIBXSMM, this value should never be used to set arch, only for compares */

LIBXSMM_EXTERN_C typedef struct LIBXSMM_RETARGETABLE libxsmm_cpuid_x86_info {
  int constant_tsc;
} libxsmm_cpuid_x86_info;

/** Returns the target architecture and instruction set extensions. */
LIBXSMM_API int libxsmm_cpuid_x86(libxsmm_cpuid_x86_info* info);

/**
 * Similar to libxsmm_cpuid_x86, but conceptually not x86-specific.
 * The actual code path (as used by LIBXSMM) is determined by
 * libxsmm_[get|set]_target_archid/libxsmm_[get|set]_target_arch.
 */
LIBXSMM_API int libxsmm_cpuid(void);

/** Names the CPU architecture given by CPUID. */
LIBXSMM_API const char* libxsmm_cpuid_name(int id);

/** SIMD vector length (VLEN) in 32-bit elements. */
LIBXSMM_API int libxsmm_cpuid_vlen32(int id);

#endif /*LIBXSMM_CPUID_H*/

