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
#include "neural/layer_stack.h"
#include "neural/dense_layer.h"
#include "neural/twoway_layer.h"


namespace ML {


struct Augmented_Deep_Net;
struct Auto_Encoder_Stack;


/*****************************************************************************/
/* AUGMENTED_DEEP_NET_UPDATES                                                */
/*****************************************************************************/

struct Augmented_Deep_Net_Updates {
    Augmented_Deep_Net_Updates();
    Augmented_Deep_Net_Updates(const Augmented_Deep_Net & net);

    ML::Parameters_Copy<double> dnae, supervised;

    Augmented_Deep_Net_Updates &
    operator += (const Augmented_Deep_Net_Updates & updates);
};


/*****************************************************************************/
/* AUGMENTED_DEEP_NET                                                        */
/*****************************************************************************/

/** This is a neural network in the following architecture:

                        models
                       iiiiiiiiii
                       oooooooooo
                       oooooooooo
                       oooooooooo
           features      oooooo
           iiiiiiii       oooo
           oooooooo       oooo
              oooooooooooooo
                     o

    There are two networks stuck together: a denoising autoencoder that is
    used to model the data, and a supervised network that takes the reduced
    representation from the denoising autoencoder and augments it with other
    features.
*/

struct Augmented_Deep_Net {
    Augmented_Deep_Net();

    void init(const ML::Auto_Encoder_Stack & dnae,
              const distribution<double> & means,
              int nfeatures,
              int nhidden, int noutputs, Transfer_Function_Type transfer,
              Thread_Context & context,
              Target target);

    ML::Layer_Stack<ML::Twoway_Layer> dnae;
    ML::Layer_Stack<ML::Twoway_Layer> supervised;
    distribution<double> means;
    Target target;

    float predict(const ML::distribution<float> & models,
                  const distribution<float> & features) const;

    void update(const Augmented_Deep_Net_Updates & updates,
                double learning_rate);

    std::pair<double, double>
    train_example(const distribution<float> & model_outpus,
                  const distribution<float> & extra_features,
                  float label,
                  Augmented_Deep_Net_Updates & updates) const;

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

};

}  // namespace ML


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

    virtual float predict(const ML::distribution<float> & models) const;

    virtual std::string explain(const ML::distribution<float> & models) const;

    /* Add in some extra features to help the classifier along */
    distribution<float>
    get_extra_features(const distribution<float> & model_outputs,
                       const distribution<float> & target_singular,
                       const Target_Stats & stats) const;

    ML::Augmented_Deep_Net net;

    bool use_extra_features;

    ML::Configuration config;
    std::vector<int> recomposition_sizes;

    std::string model_base;
    int random_seed;

    const Data * data;
    Target target;
};

#endif /* __ausdm__deep_net_blender_h__ */

