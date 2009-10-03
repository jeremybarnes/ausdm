/* gated_blender.cc
   Jeremy Barnes, 30 September 2009
   Copyright (c) 2009 Jeremy Barnes.  All rights reserved.

   Blender that uses gated.
*/

#include "gated_blender.h"
#include "utils/vector_utils.h"
#include "algebra/irls.h"
#include "boosting/worker_task.h"
#include "utils/guard.h"
#include <boost/bind.hpp>


using namespace ML;
using namespace std;


/*****************************************************************************/
/* GATED_BLENDER                                                             */
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
train_model(int model, const Data & training_data)
{
    // Generate a matrix with the predictions
    typedef float Float;

    int nx = training_data.targets.size();
    int nv = get_model_features
        (model,
         distribution<float>(training_data.models.size()),
         distribution<double>(training_data.singular_values.size()))
        .size();


    // Assemble the labels
    distribution<float> correct(nx);
    boost::multi_array<float, 2> outputs(boost::extents[nv][nx]);
    distribution<float> w(nx, 1.0 / nx);

    for (unsigned i = 0;  i < training_data.targets.size();  ++i) {

        // For now, we will try to predict if the margin > 0.  This only
        // works for AUC; for RMSE we will need something different.
        // Eventually, we might want to predict the margin directly or
        // take a threshold for the margin, eg 0.5
        float pred = (training_data.models[model][i] - 3.0) / 2.0;
        float margin = pred * training_data.targets[i];

        correct[i] = (margin >= 0.0);
        //correct[i] = training_data.targets[i] > 0.0;

        distribution<float> model_outputs(training_data.models.size());
        for (unsigned j = 0;  j < training_data.models.size();  ++j)
            model_outputs[j] = training_data.models[j][i];
        
        distribution<double> target_singular
            (training_data.singular_targets[i].begin(),
             training_data.singular_targets[i].end());

        distribution<float> features
            = get_model_features(model, model_outputs, target_singular);

        if (features.size() != nv)
            throw Exception("nv is wrong");

        for (unsigned j = 0;  j < nv;  ++j)
            outputs[j][i] = features[j];
    }

    distribution<float> parameters
        = irls_logit(correct, outputs, w);
    
    cerr << "parameters for model " << model << ": " << parameters << endl;

    model_coefficients[model] = parameters;
}

void
Gated_Blender::
init(const Data & training_data)
{
    this->data = &training_data;

    // Now to train.  For each of the models, we go through the training
    // data and create a data file; we then do an IRLS on the model.

    int num_models_to_train = 10;

    model_coefficients.resize(training_data.models.size());


    static Worker_Task & worker = Worker_Task::instance(num_threads() - 1);
        
    // Now, submit it as jobs to the worker task to be done multithreaded
    int group;
    {
        int parent = -1;  // no parent group
        group = worker.get_group(NO_JOB, "train model task", parent);
            
        // Make sure the group gets unlocked once we've populated
        // everything
        Call_Guard guard(boost::bind(&Worker_Task::unlock_group,
                                     boost::ref(worker),
                                     group));

        for (unsigned i = 0;  i < training_data.models.size();  ++i) {
            if (training_data.models[i].rank >= num_models_to_train) continue;

            worker.add(boost::bind(&Gated_Blender::train_model,
                                   this,
                                   i,
                                   boost::cref(training_data)),
                       "train model job",
                       group);
        }
    }

    // Add this thread to the thread pool until we're ready
    worker.run_until_finished(group);
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
get_model_features(int model,
                   const distribution<float> & model_outputs,
                   const distribution<double> & target_singular) const
{
    // Features:
    // 1.  The current model's output
    // 2.  Target singular values
    // 3.  Error with 10 models
    // 4.  Error with 50 models

    float real_prediction = model_outputs[model];
    
    distribution<float> weights(data->singular_values.size(), 0.0);
    for (unsigned j = 0;  j < 10;  ++j)
        weights[j] = 1.0;
    
    float model_prediction_10
        = (target_singular * data->singular_values)
        .dotprod(data->singular_models[model] * weights);
    
    for (unsigned j = 10;  j < 50;  ++j)
        weights[j] = 1.0;
    
    float model_prediction_50
        = (target_singular * data->singular_values)
        .dotprod(data->singular_models[model] * weights);

    distribution<float> result;

    result.push_back(real_prediction);
    result.insert(result.end(),
                  target_singular.begin(), target_singular.end());
    result.push_back(model_prediction_10 - real_prediction);
    result.push_back(fabs(result.back()));
    result.push_back(model_prediction_50 - real_prediction);
    result.push_back(fabs(result.back()));

    return result;
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
        // Skip untrained models
        if (model_coefficients[i].empty()) continue;

        // What would we have predicted for this model?

        distribution<float> model_features
            = get_model_features(i, models, target_singular);

        // Perform linear regression (in prediction mode)
        float output = model_features.dotprod(model_coefficients[i]);

        // Link function to change into a probability
        float prob = apply_link_inverse(output, LOGIT);

        result[i] = prob;
    }

    return result;
}

float
Gated_Blender::
predict(const ML::distribution<float> & models) const
{
    bool debug = false;

    if (debug) {
        static Lock lock;
        Guard guard(lock);
    }

    distribution<float> conf = this->conf(models);
    
    for (unsigned i = 0;  i < models.size() && debug;  ++i) {
        if (conf[i] == 0.0) continue;
        cerr << "model " << i << ": pred " << models[i] << " conf "
             << conf[i] << endl;
    }

    distribution<float> model_preds(models.size());
    for (unsigned i = 0;  i < models.size();  ++i)
        model_preds[i] = (models[i] > 3.0 ? 1.0 : -1.0);

    //float result = model_preds.dotprod(conf) / conf.total();

    float result = models.dotprod(conf) / conf.total();

    //float result = conf.total() * 0.1 * 4.0 + 1.0;

    if (debug) cerr << "result = " << result << endl;

    return result;
}
