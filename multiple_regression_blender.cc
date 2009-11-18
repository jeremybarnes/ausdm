/* multiple_regression_blender.cc
   Jeremy Barnes, 2 November 2009
   Copyright (c) 2009 Jeremy Barnes.  All rights reserved.

   Blender for multiple regression models.
*/

#include "multiple_regression_blender.h"
#include "utils.h"


using namespace std;
using namespace ML;


/*****************************************************************************/
/* MULTIPLE_REGRESSION_BLENDER                                               */
/*****************************************************************************/

Multiple_Regression_Blender::
Multiple_Regression_Blender()
{
}

Multiple_Regression_Blender::
~Multiple_Regression_Blender()
{
}

void
Multiple_Regression_Blender::
configure(const ML::Configuration & config_,
          const std::string & name,
          int random_seed,
          Target target)
{
    Configuration config(config_, name, Configuration::PREFIX_APPEND);
    
    //config.require(link_function, "link_function");

    link_function = (target == AUC ? LINEAR : LOGIT);

    this->random_seed = random_seed;
    this->target = target;
}

distribution<float>
Multiple_Regression_Blender::
train_model(const Data & data,
            Thread_Context & thread_context) const
{
    // Number of examples to take
    int nx_me = 5000;

    // Number of models to take
    int nm_me = 100;
    
    int nx = data.nx();
    int nm = data.nm();

    // Choose models randomly
    set<int> models_done;
    while (models_done.size() < nm_me) {
        models_done.insert(thread_context.random01() * nm);
    }

    vector<int> models(models_done.begin(), models_done.end());
    

    // Choose examples randomly
    set<int> examples_done;
    while (examples_done.size() < nx_me) {
        examples_done.insert(thread_context.random01() * nx);
    }

    vector<int> examples(examples_done.begin(), examples_done.end());

    typedef double Float;
    distribution<Float> correct(nx_me);
    boost::multi_array<Float, 2> outputs(boost::extents[nm_me][nx_me]);
    distribution<Float> w(nx_me, 1.0);

    for (unsigned i = 0;  i < nx_me;  ++i) {
        distribution<float> features = data.examples[examples[i]]->models;

        for (unsigned j = 0;  j < nm_me;  ++j)
            outputs[j][i] = features[models[j]];

        correct[i] = data.targets[examples[i]];
        if (target == AUC)
            correct[i] = (correct[i] == 1.0);
    }
    
    distribution<Float> trained_params
        = perform_irls(correct, outputs, w, link_function);

    distribution<float> result(nm);

    for (unsigned i = 0;  i < models.size();  ++i)
        result[models[i]] = trained_params[i];

    return result;
}

void
Multiple_Regression_Blender::
init(const Data & training_data,
     const ML::distribution<float> & example_weights)
{
    Data train = training_data, test;

    coefficients.clear();
    coefficients.resize(train.nm());
    
    Thread_Context context;

    int num_models = 100;

    for (unsigned i = 0;  i < num_models;  ++i) {
        cerr << "i = " << i << endl;
        coefficients += train_model(train, context);
    }

    coefficients /= num_models;
}

float
Multiple_Regression_Blender::
predict(const ML::distribution<float> & models) const
{
    distribution<float> features = models;

    float output = features.dotprod(coefficients);

    // Link function to change into a probability
    float prob = apply_link_inverse(output, link_function);

    return prob;
}

std::string
Multiple_Regression_Blender::
explain(const ML::distribution<float> & models) const
{
    return "";
}

