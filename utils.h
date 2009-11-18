/* utils.h                                                         -*- C++ -*-
   Jeremy Barnes, 2 November 2009
   Copyright (c) 2009 Jeremy Barnes.  All rights reserved.

   Utility functions.
*/

#ifndef __ausdm__utils_h__
#define __ausdm__utils_h__

#include "algebra/irls.h"

namespace ML {

distribution<float>
perform_irls(const distribution<float> & correct,
             const boost::multi_array<float, 2> & outputs,
             const distribution<float> & w,
             Link_Function link_function,
             bool ridge_regression = true);

distribution<double>
perform_irls(const distribution<double> & correct,
             const boost::multi_array<double, 2> & outputs,
             const distribution<double> & w,
             Link_Function link_function,
             bool ridge_regression = true);

} // namespace ML

#endif /* __ausdm__utils_h__ */
