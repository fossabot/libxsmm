#ifndef LIBXSMM_CONFIG_H
#define LIBXSMM_CONFIG_H

#define LIBXSMM_CONFIG_VERSION "$VERSION"
#define LIBXSMM_CONFIG_BRANCH "$BRANCH"
#define LIBXSMM_CONFIG_VERSION_MAJOR $MAJOR
#define LIBXSMM_CONFIG_VERSION_MINOR $MINOR
#define LIBXSMM_CONFIG_VERSION_UPDATE $UPDATE
#define LIBXSMM_CONFIG_VERSION_PATCH $PATCH

#define LIBXSMM_CONFIG_CACHELINE $CACHELINE
#define LIBXSMM_CONFIG_ALIGNMENT $CACHELINE
#define LIBXSMM_CONFIG_ILP64 $ILP64
#define LIBXSMM_CONFIG_SYNC $SYNC
#define LIBXSMM_CONFIG_JIT $JIT

#define LIBXSMM_CONFIG_PREFETCH $PREFETCH
#define LIBXSMM_CONFIG_MAX_MNK $MAX_MNK
#define LIBXSMM_CONFIG_MAX_M $MAX_M
#define LIBXSMM_CONFIG_MAX_N $MAX_N
#define LIBXSMM_CONFIG_MAX_K $MAX_K
#define LIBXSMM_CONFIG_AVG_M $AVG_M
#define LIBXSMM_CONFIG_AVG_N $AVG_N
#define LIBXSMM_CONFIG_AVG_K $AVG_K
#define LIBXSMM_CONFIG_FLAGS $FLAGS
#define LIBXSMM_CONFIG_ALPHA $ALPHA
#define LIBXSMM_CONFIG_BETA $BETA
#define LIBXSMM_CONFIG_WRAP $WRAP
$LIBXSMM_OFFLOAD_BUILD

$MNK_PREPROCESSOR_LIST

#endif
