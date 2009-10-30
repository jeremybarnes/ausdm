/* deep_net_blender.cc
   Jeremy Barnes, 30 October 2009
   Copyright (c) 2009 Jeremy Barnes.  All rights reserved.

   Blender based upon training of a deep neural network.
*/

#include "deep_net_blender.h"


using namespace std;
using namespace ML;


/*****************************************************************************/
/* DEEP_NET_BLENDER                                                          */
/*****************************************************************************/

Deep_Net_Blender::
Deep_Net_Blender()
{
}

Deep_Net_Blender::
~Deep_Net_Blender()
{
}

void
Deep_Net_Blender::
configure(const ML::Configuration & config,
          const std::string & name,
          int random_seed,
          Target target)
{
    this->config = config;
    this->random_seed = random_seed;

    config.require(model_base, "model_base");
}

void
Deep_Net_Blender::
init(const Data & data,
     const ML::distribution<float> & example_weights)
{
    // Reconstitute the base model
    {
        boost::shared_ptr<Decomposition> loaded;
        const DNAE_Decomposition & decomp
            = dynamic_cast<const DNAE_Decomposition &>
                 (*(loaded = Decomposition::load(model_base)));
        stack = decomp.stack;
    }

    Thread_Context context;
    context.seed(random_seed);

    // Add a new layer
    Twoway_Layer top_layer(false /* use_dense_missing */,
                           stack.back().outputs() /* inputs */,
                           1 /* outputs */,
                           TF_TANH,
                           context);

    stack.push_back(top_layer);

    Data training_data = data;
    Data testing_data;
    training_data.hold_out(testing_data, 0.5, random_seed);

    // Start training
    vector<distribution<float> > training_samples = training_data.examples;
    vector<distribution<float> > testing_samples = testing_data.examples;

    int nx = training_samples.size();

    for (unsigned i = 0;  i < nx;  ++i)
        training_samples[i] *= 0.8;

    int nxt = testing_samples.size();

    for (unsigned i = 0;  i < nxt;  ++i)
        testing_samples[i] *= 0.8;

    stack.train_discrim(training_samples, training_data.targets,
                        testing_samples, testing_data.targets,
                        config, context);
}

float
Deep_Net_Blender::
predict(const ML::distribution<float> & models) const
{
    return 1.25 * stack.apply(0.8 * models)[0];
}

std::string
Deep_Net_Blender::
explain(const ML::distribution<float> & models) const
{
    return "no explanation";
}

