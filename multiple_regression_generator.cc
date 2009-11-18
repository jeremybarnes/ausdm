/* multiple_regression_generator.cc
   Jeremy Barnes, 2 November 2009
   Copyright (c) 2009 Jeremy Barnes.  All rights reserved.

   Generator for multiple regression models.
*/

#include "multiple_regression_generator.h"

#if 0

/*****************************************************************************/
/* MULTIPLE_REGRESSION_BLENDER                                               */
/*****************************************************************************/

Muiltiple_Regression_Blender::
Multiple_Regression_Blender()
{
}

Muiltiple_Regression_Blender::
~Multiple_Regression_Blender()
{
}

void
Muiltiple_Regression_Blender::
configure(const ML::Configuration & config,
          const std::string & name,
          int random_seed,
          Target target)
{
}

void
Multiple_Regression_Blender::
train_model()
{
    // Number of examples to take
    int nx_me = 3000;

    // Number of models to take
    int nm_me = 100;

    set<int> models_done;
    while (models_done.size() < nm_me) {
        models_done.insert(thread_context.random01() * nm);
    }

    vector<int> models(models_done.begin(), models_done.end());
    
    set<int> examples_done;
    while (examples_done.size() < nx_me) {
        examples_done.insert(thread_context.random01() * nm);
    }

    vector<int> examples(examples_done.begin(), examples_done.end());

    typedef double Float;
    distribution<Float> correct(nx_me);
    boost::multi_array<Float, 2> outputs(boost::extents[nm_me][nx_me]);
    distribution<Float> w(nx_me, 1.0);

    for (unsigned j = 0;  j < nm_me;  ++j) {
        for (unsigned i = 0;  i < nx_me;  ++i) {
            outputs[j] = features[j] * 
                                  }
    }
    int i = 0;
    for (set<int>::const_iterator
             it = examples_done.begin(),
             end = examples_done.end();
         it != end;  ++it) {
        //for (set<int>::const_iterator jt = it->begin
    }

    float p_in = min(1.0, 2.0 / n_irls);
    
    vector<int> examples;
    for (unsigned x = 0;  x < nx;  ++x) {
        if (context.random01() >= p_in) continue;
        examples.push_back(x);
    }

    int nx2 = examples.size();
    
    for (unsigned v = 0;  v < nv;  ++v)
        for (unsigned i = 0;  i < nx2;  ++i)
            outputs2[v][i] = outputs[v][examples[i]];
    
    for (unsigned i = 0;  i < nx2;  ++i) {
        int x = examples[i];
        correct2[i] = correct[x];
        w2[i] = w[x];
    }
    
    distribution<Float> trained_params
        = perform_irls(correct2, outputs2, w2, link_function);
}

void
Muiltiple_Regression_Blender::
init(const Data & training_data,
     const ML::distribution<float> & example_weights)
{
    typedef double Float;

    // Assemble the labels
    distribution<Float> correct(nx);
    boost::multi_array<Float, 2> outputs(boost::extents[nv][nx]);
    distribution<Float> w(example_weights.begin(), example_weights.end());

    for (unsigned i = 0;  i < training_data.nx();  ++i) {

        const distribution<float> & model_outputs
            = training_data.examples[i]->models;

        const distribution<float> & target_singular
            = training_data.examples[i]->decomposed;

        distribution<float> features
            = get_features(model, model_outputs, target_singular,
                           training_data.examples[i]->stats);

        //cerr << "conf features: " << features << endl;
        
        if (features.size() != nv)
            throw Exception("nv is wrong");

        for (unsigned j = 0;  j < nv;  ++j)
            outputs[j][i] = features[j];
    }

}

float
Muiltiple_Regression_Blender::
predict(const ML::distribution<float> & models,
        const distribution<float> & features) const
{
}

float
Muiltiple_Regression_Blender::
predict(const ML::distribution<float> & models) const
{
    distribution<float> features = get_features(models);

    float output = model_features.dotprod(coefficients);

    // Link function to change into a probability
    float prob = apply_link_inverse(output, link_function);

    return prob;
}

std::string
Muiltiple_Regression_Blender::
explain(const ML::distribution<float> & models) const
{
    return "";
}

distribution<float>
Muiltiple_Regression_Blender::
get_features(const distribution<float> & model_outputs,
             const distribution<float> & target_singular,
             const Target_Stats & stats) const
{
    return model_outputs;
}

distribution<double>
Muiltiple_Regression_Blender::
train_model(const std::vector<distribution<float> > & model_outputs,
            const std::vector<distribution<float> > & extra_features,
            const std::vector<float> & labels,
            ML::Thread_Context & thread_context) const
{
}

#endif
