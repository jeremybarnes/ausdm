/* multiple_regression_generator.h                                 -*- C++ -*-
   Jeremy Barnes, 1 November 2009
   Copyright (c) 2009 Jeremy Barnes.  All rights reserved.

   Perform regression multiple times.
*/

#ifndef __ausdm__multiple_regression_generator_h__
#define __ausdm__multiple_regerssion_generator_h__

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
    
    virtual void init(const Data & training_data,
                      const ML::distribution<float> & example_weights);

    float predict(const ML::distribution<float> & models,
                  const distribution<float> & features) const;

    virtual float predict(const ML::distribution<float> & models) const;

    virtual std::string explain(const ML::distribution<float> & models) const;

    /* Add in some extra features to help the classifier along */
    distribution<float>
    get_extra_features(const distribution<float> & model_outputs,
                       const distribution<float> & target_singular,
                       const Target_Stats & stats) const;


    /** Trains a single iteration on the given data with the selected
        parameters.  Returns a moving estimate of the RMSE on the
        training set. */
    distribution<double>
    train_model(const std::vector<distribution<float> > & model_outputs,
                const std::vector<distribution<float> > & extra_features,
                const std::vector<float> & labels,
                ML::Thread_Context & thread_context) const;

    /** Test the discriminative power of the network.  Returns the RMS error
        or AUC depending upon whether it's a regression or classification
        task.
    */
    std::pair<double, double>
    test(const std::vector<distribution<float> > & model_outputs,
         const std::vector<distribution<float> > & extra_features,
         const std::vector<float> & labels,
         ML::Thread_Context & thread_context,
         int verbosity);
    int random_seed;

    distribution<double> params;

    const Data * data;
};

#endif /* __ausdm__multiple_regression_blender_h__ */


