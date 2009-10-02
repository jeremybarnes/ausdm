/* gated_blender.cc
   Jeremy Barnes, 30 September 2009
   Copyright (c) 2009 Jeremy Barnes.  All rights reserved.

   Blender that uses gated.
*/

#include "gated_blender.h"
#include "utils/vector_utils.h"


using namespace ML;
using namespace std;


/*****************************************************************************/
/* GATED_BLENDER                                                          */
/*****************************************************************************/

Gated_Blender::Gated_Blender()
    : data(0)
{
}

Gated_Blender::~Gated_Blender()
{
}
    
void
Gated_Blender::
configure(const ML::Configuration & config,
          const std::string & name,
          int random_seed)
{
}
    
void
Gated_Blender::
init(const Data & training_data)
{
    this->data = &training_data;
}

boost::shared_ptr<Dense_Feature_Space>
Gated_Blender::
conf_feature_space() const
{
    boost::shared_ptr<Dense_Feature_Space> result
        (new Dense_Feature_Space());

    return result;
    // Features: output, nmodels principal components
}

distribution<float>
Gated_Blender::
conf(const ML::distribution<float> & models) const
{
    // First, get the singular vector for the model
    distribution<double> target_singular(data->singular_values.size());

    for (unsigned i = 0;  i < models.size();  ++i)
        target_singular += data->singular_models[i] * models[i];
    
    target_singular /= data->singular_values;

    // For each model, calculate a confidence
    distribution<float> result(models.size());

    for (unsigned i = 0;  i < models.size();  ++i) {
        // What would we have predicted for this model?

        float model_prediction
            = (target_singular * data->singular_values)
            .dotprod(data->singular_models[i]);

        float real_prediction
            = models[i];

        cerr << "model " << i << ": model_prediction = "
             << model_prediction << " real_prediction = "
             << real_prediction << endl;

        result[i] = 1.0;
    }

    result.normalize();

    return result;
}

float
Gated_Blender::
predict(const ML::distribution<float> & models) const
{
    distribution<float> conf = this->conf(models);
    
    return models.dotprod(conf) / conf.total();
}
