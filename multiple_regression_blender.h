/* multiple_regression_blender.h                                 -*- C++ -*-
   Jeremy Barnes, 1 November 2009
   Copyright (c) 2009 Jeremy Barnes.  All rights reserved.

   Perform regression multiple times.
*/

#ifndef __ausdm__multiple_regression_blender_h__
#define __ausdm__multiple_regerssion_blender_h__

#include "blender.h"
#include "boosting/dense_features.h"
#include "boosting/classifier.h"
#include "algebra/irls.h"
#include "utils/filter_streams.h"
#include "dnae_decomposition.h"


/*****************************************************************************/
/* MULTIPLE_REGRESSION_BLENDER                                               */
/*****************************************************************************/

struct Multiple_Regression_Blender : public Blender {

    Multiple_Regression_Blender();

    virtual ~Multiple_Regression_Blender();

    virtual void configure(const ML::Configuration & config,
                           const std::string & name,
                           int random_seed,
                           Target target);
    
    distribution<float>
    train_model(const Data & data,
                ML::Thread_Context & thread_context) const;

    virtual void init(const Data & training_data,
                      const ML::distribution<float> & example_weights);

    virtual float predict(const ML::distribution<float> & models) const;

    virtual std::string explain(const ML::distribution<float> & models) const;

    int random_seed;
    int num_iter;
    int num_examples;
    int num_features;
    bool ridge_regression;

    distribution<double> coefficients;

    ML::Link_Function link_function;

    Target target;
};

#endif /* __ausdm__multiple_regression_blender_h__ */


