#ifndef __c2_sl_puma_hw6_h__
#define __c2_sl_puma_hw6_h__

/* Include files */
#include "sf_runtime/sfc_sf.h"
#include "sf_runtime/sfc_mex.h"
#include "rtwtypes.h"
#include "multiword_types.h"

/* Type Definitions */
#ifndef typedef_SFc2_sl_puma_hw6InstanceStruct
#define typedef_SFc2_sl_puma_hw6InstanceStruct

typedef struct {
  SimStruct *S;
  ChartInfoStruct chartInfo;
  uint32_T chartNumber;
  uint32_T instanceNumber;
  int32_T c2_sfEvent;
  boolean_T c2_isStable;
  boolean_T c2_doneDoubleBufferReInit;
  uint8_T c2_is_active_c2_sl_puma_hw6;
  real_T *c2_t;
  real_T (*c2_y)[6];
  real_T (*c2_time)[1001];
  real_T (*c2_torque)[6006];
} SFc2_sl_puma_hw6InstanceStruct;

#endif                                 /*typedef_SFc2_sl_puma_hw6InstanceStruct*/

/* Named Constants */

/* Variable Declarations */
extern struct SfDebugInstanceStruct *sfGlobalDebugInstanceStruct;

/* Variable Definitions */

/* Function Declarations */
extern const mxArray *sf_c2_sl_puma_hw6_get_eml_resolved_functions_info(void);

/* Function Definitions */
extern void sf_c2_sl_puma_hw6_get_check_sum(mxArray *plhs[]);
extern void c2_sl_puma_hw6_method_dispatcher(SimStruct *S, int_T method, void
  *data);

#endif
