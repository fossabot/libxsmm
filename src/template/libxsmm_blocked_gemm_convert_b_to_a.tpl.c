/******************************************************************************
* Copyright (c) Intel Corporation - All rights reserved.                      *
* This file is part of the LIBXSMM library.                                   *
*                                                                             *
* For information on the license, see the LICENSE file.                       *
* Further information: https://github.com/hfp/libxsmm/                        *
* SPDX-License-Identifier: BSD-3-Clause                                       *
******************************************************************************/
/* Kunal Banerjee (Intel Corp.)
******************************************************************************/

LIBXSMM_VLA_DECL(4, LIBXSMM_BLOCKED_GEMM_TEMPLATE_TYPE, real_dst, (LIBXSMM_BLOCKED_GEMM_TEMPLATE_TYPE*)dst, handle->nb, handle->bn, handle->bm);
LIBXSMM_VLA_DECL(4, const LIBXSMM_BLOCKED_GEMM_TEMPLATE_TYPE, real_src, (const LIBXSMM_BLOCKED_GEMM_TEMPLATE_TYPE*)src, handle->mb, handle->bn, handle->bm);

libxsmm_blasint mb, nb, bm, bn;

for (mb = 0; mb < handle->mb; ++mb) {
  for (nb = 0; nb < handle->nb; ++nb) {
    for (bn = 0; bn < handle->bn; ++bn) {
      for (bm = 0; bm < handle->bm; ++bm) {
        LIBXSMM_VLA_ACCESS(4, real_dst, mb, nb, bn, bm, handle->nb, handle->bn, handle->bm) =
        LIBXSMM_VLA_ACCESS(4, real_src, nb, mb, bn, bm, handle->mb, handle->bn, handle->bm);
      }
    }
  }
}
