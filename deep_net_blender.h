/* deep_net_blender.h                                              -*- C++ -*-
   Jeremy Barnes, 30 September 2009
   Copyright (c) 2009 Jeremy Barnes.  All rights reserved.

   Blender that trains on a deep neural network.
*/

#ifndef __ausdm__deep_net_blender_h__
#define __ausdm__deep_net_blender_h__


#include "blender.h"
#include "boosting/dense_features.h"
#include "boosting/classifier.h"
#include "algebra/irls.h"
#include "utils/filter_streams.h"
#include "dnae_decomposition.h"


/*****************************************************************************/
/* DEEP_NET_BLENDER                                                          */
/*****************************************************************************/

struct Deep_Net_Blender : public Blender {

    Deep_Net_Blender();

    virtual ~Deep_Net_Blender();

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


    std::pair<double, double>
    train_example(const distribution<float> & model_outpus,
                  const distribution<float> & extra_features,
                  float label,
                  ML::DNAE_Stack_Updates & dnae_updates,
                  ML::DNAE_Stack_Updates & supervised_updates) const;

    /** Trains a single iteration on the given data with the selected
        parameters.  Returns a moving estimate of the RMSE on the
        training set. */
    std::pair<double, double>
    train_iter(const std::vector<distribution<float> > & model_outputs,
               const std::vector<distribution<float> > & extra_features,
               const std::vector<float> & labels,
               ML::Thread_Context & thread_context,
               int minibatch_size, float learning_rate,
               int verbosity,
               float sample_proportion,
               bool randomize_order);

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

    ML::Configuration config;
    std::vector<int> recomposition_sizes;

    std::string model_base;
    int random_seed;

    ML::DNAE_Stack dnae_stack;
    ML::DNAE_Stack supervised_stack;

    const Data * data;
};

#endif /* __ausdm__deep_net_blender_h__ */

