/* blender.cc
   Jeremy Barnes, 30 September 2009
   Copyright (c) 2009 Jeremy Barnes.  All rights reserved.

   Implementation of the blender class.
*/


#include "blender.h"
#include "boosting_blender.h"
#include "gated_blender.h"


using namespace ML;
using namespace std;

__thread float correct_prediction = 0.0;


/*****************************************************************************/
/* BLENDER                                                                   */
/*****************************************************************************/

Blender::
Blender()
{
}

Blender::
~Blender()
{
}

std::string
Blender::
explain(const ML::distribution<float> & models) const
{
    return "";
}


/*****************************************************************************/
/* LINEAR_BLENDER                                                            */
/*****************************************************************************/

Linear_Blender::
Linear_Blender()
{
}

Linear_Blender::
~Linear_Blender()
{
}

void
Linear_Blender::
configure(const ML::Configuration & config_,
          const std::string & name,
          int random_seed,
          Target target)
{
    Configuration config(config_, name, Configuration::PREFIX_APPEND);
    
    config.require(mode, "mode");
    config.require(num_models, "num_models");
}

void
Linear_Blender::
init(const Data & data,
     const ML::distribution<float> & example_weights)
{
    model_weights.clear();
    model_weights.resize(data.models.size(), 0.0);

    if (mode == "best_n" || mode == "best_n_weighted") {
        if (num_models > data.models.size())
            num_models = data.models.size();

        for (unsigned i = 0;  i < num_models;  ++i) {
            int m = data.model_ranking[i];
            float score
                = (mode == "best_n_weighted")
                ? data.models[m].score
                : 1.0;

            model_weights[m] = score;
        }

        model_weights.normalize();
    }
    else throw Exception("unknown mode " + mode);
}

float
Linear_Blender::
predict(const ML::distribution<float> & models) const
{
    return models.dotprod(model_weights);
}


/*****************************************************************************/
/* UTILITY FUNCTIONS                                                         */
/*****************************************************************************/

boost::shared_ptr<Blender>
get_blender(const ML::Configuration & config_,
            const std::string & name,
            const Data & data,
            const ML::distribution<float> & example_weights,
            int random_seed,
            Target target)
{
    Configuration config(config_, name, Configuration::PREFIX_APPEND);

    string type;
    config.require(type, "type");

    boost::shared_ptr<Blender> result;

    if (type == "linear")
        result.reset(new Linear_Blender());
    else if (type == "boosting")
        result.reset(new Boosting_Blender());
    else if (type == "gated")
        result.reset(new Gated_Blender());
    else throw Exception("Blender of type " + type + " doesn't exist");

    result->configure(config_, name, random_seed, target);
    result->init(data, example_weights);
    
    return result;
}

