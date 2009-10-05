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
#include "algebra/lapack.h"


using namespace ML;
using namespace std;


/*****************************************************************************/
/* GATED_BLENDER                                                             */
/*****************************************************************************/

Gated_Blender::Gated_Blender()
    : link_function(LOGIT), data(0)
{
}

Gated_Blender::~Gated_Blender()
{
}
    
void
Gated_Blender::
configure(const ML::Configuration & config_,
          const std::string & name,
          int random_seed,
          Target target)
{
    Configuration config(config_, name, Configuration::PREFIX_APPEND);
    
    config.require(link_function, "link_function");

    debug_predict = false;
    config.find(debug_predict, "debug_predict");

    config.require(num_models_to_train, "num_models_to_train");

    this->target = target;
}

void
Gated_Blender::
train_conf(int model, const Data & training_data)
{
    // Generate a matrix with the predictions
    int nx = training_data.targets.size();
    int nv = get_model_features
        (model,
         distribution<float>(training_data.models.size()),
         distribution<double>(training_data.singular_values.size()),
         Target_Stats())
        .size();


    typedef double Float;

    // Assemble the labels
    distribution<Float> correct(nx);
    boost::multi_array<Float, 2> outputs(boost::extents[nv][nx]);
    distribution<Float> w(nx, 1.0 / nx);

    for (unsigned i = 0;  i < training_data.targets.size();  ++i) {

        if (target == AUC) {
            // For now, we will try to predict if the margin > 0.  This only
            // works for AUC; for RMSE we will need something different.
            // Eventually, we might want to predict the margin directly or
            // take a threshold for the margin, eg 0.5
            float pred = (training_data.models[model][i] - 3.0) / 2.0;
            float margin = pred * training_data.targets[i];
            
            correct[i] = (margin >= 0.0);
            //correct[i] = training_data.targets[i] > 0.0;
            //correct[i] = (margin * 0.5) + 0.5;
        }
        else {
            // Try to predict the probability that it's within 0.5 either side
            // of the correct answer
            correct[i]
                = abs(training_data.models[model][i]
                      - training_data.targets[i]) <= 0.5;
        }

        distribution<float> model_outputs(training_data.models.size());
        for (unsigned j = 0;  j < training_data.models.size();  ++j)
            model_outputs[j] = training_data.models[j][i];
        
        distribution<double> target_singular
            (training_data.singular_targets[i].begin(),
             training_data.singular_targets[i].end());

        distribution<float> features
            = get_model_features(model, model_outputs, target_singular,
                                 training_data.target_stats[i]);

        if (features.size() != nv)
            throw Exception("nv is wrong");

        for (unsigned j = 0;  j < nv;  ++j)
            outputs[j][i] = features[j];
    }

    distribution<double> svalues1
        (std::min(outputs.shape()[0], outputs.shape()[1]));

    int result = LAPack::gesdd("N", outputs.shape()[1], outputs.shape()[0],
                               outputs.data(), outputs.shape()[1], 
                               &svalues1[0], 0, 1, 0, 1);

    if (result != 0)
        throw Exception("error in SVD");

#if 0
    cerr << "result of svd = " << result << endl;
    
    cerr << "svalues = " << svalues << endl;
#endif


    // Remove linearly dependent columns.  Returns a vector saying in which
    // entry of the new matrix the current column really is
    vector<int> new_loc = remove_dependent(outputs);

    distribution<Float> trained
        = run_irls(correct, outputs, w, link_function);

    distribution<float> parameters(nv);
    for (unsigned v = 0;  v < nv;  ++v)
        if (new_loc[v] != -1) parameters[v] = trained[new_loc[v]];
    
    //cerr << "parameters for model " << model << ": " << parameters << endl;

    Model_Output before, after;
    before.resize(training_data.targets.size());
    after.resize(training_data.targets.size());

    // Test the original model and the weighted version for AUC
    for (unsigned i = 0;  i < training_data.targets.size();  ++i) {

        distribution<float> features(nv);
        for (unsigned j = 0;  j < nv;  ++j)
            if (new_loc[j] != -1)
                features[j] = outputs[new_loc[j]][i];

        float result = apply_link_inverse(features.dotprod(parameters),
                                          link_function);

        before[i] = features[0];
        after[i] = result;
    }

    //cerr << "before = " << before << endl;

    float auc_before1 = before.calc_score(training_data.targets, target);
    float auc_after1  = after.calc_score(training_data.targets, target);
    float auc_before2 = before.calc_auc(correct * 2.0 - 1.0);
    float auc_after2  = after.calc_auc(correct * 2.0 - 1.0);


    static Lock lock;
    Guard guard(lock);
    
    cerr << "model " << model
         << ": before " << auc_before1 << "/" << auc_before2
         << " after " << auc_after1 << "/" << auc_after2 << endl;


    if (auc_after2 == 1.0) {
        cerr << "new_loc = " << new_loc << endl;
        cerr << "parameters = " << parameters << endl;
        cerr << "trained = " << trained << endl;
        //cerr << "after = " << after << endl;

        distribution<double> svalues
            (std::min(outputs.shape()[0], outputs.shape()[1]));

        int result = LAPack::gesdd("N", outputs.shape()[1], outputs.shape()[0],
                                   outputs.data(), outputs.shape()[1], 
                                   &svalues[0], 0, 1, 0, 1);

        cerr << "result of svd = " << result << endl;

        cerr << "svalues = " << svalues << endl;
        cerr << "svalues1 = " << svalues1 << endl;
        

        throw Exception("something wrong on AUC 2");
    }

    model_coefficients[model] = parameters;
}

void
Gated_Blender::
init(const Data & training_data_in)
{
    this->data = &training_data_in;

    Data conf_training_data = training_data_in;
    Data blend_training_data;

    conf_training_data.hold_out(blend_training_data, 0.50);

    conf_training_data.stats();

    // Now to train.  For each of the models, we go through the training
    // data and create a data file; we then do an IRLS on the model.

    model_coefficients.resize(conf_training_data.models.size());


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

        for (unsigned i = 0;  i < conf_training_data.models.size();  ++i) {
            if (training_data_in.models[i].rank >= num_models_to_train)
                continue;

            worker.add(boost::bind(&Gated_Blender::train_conf,
                                   this, i, boost::cref(conf_training_data)),
                       "train model job",
                       group);
        }
    }

    // Add this thread to the thread pool until we're ready
    worker.run_until_finished(group);


    int nx = blend_training_data.targets.size();

    int nv = get_blend_features
        (distribution<float>(blend_training_data.models.size()),
         distribution<float>(blend_training_data.models.size()))
        .size();


    cerr << "generating blend data" << endl;

    typedef double BlendFloat;

    // Assemble the labels
    distribution<BlendFloat> correct(nx);
    boost::multi_array<BlendFloat, 2> outputs(boost::extents[nv][nx]);
    distribution<BlendFloat> w(nx, 1.0 / nx);

    for (unsigned i = 0;  i < blend_training_data.targets.size();  ++i) {

        if (target == AUC)
            correct[i] = blend_training_data.targets[i] > 0.0;
        else
            correct[i] = blend_training_data.targets[i];

        correct_prediction = correct[i];

        distribution<float> model_outputs(blend_training_data.models.size());
        for (unsigned j = 0;  j < blend_training_data.models.size();  ++j)
            model_outputs[j] = blend_training_data.models[j][i];
        
        distribution<double> target_singular
            (blend_training_data.singular_targets[i].begin(),
             blend_training_data.singular_targets[i].end());

        distribution<float> conf = this->conf(model_outputs);

        distribution<float> features
            = get_blend_features(model_outputs, conf);

        if (features.size() != nv)
            throw Exception("nv is wrong");

        for (unsigned j = 0;  j < nv;  ++j)
            outputs[j][i] = features[j];
    }

    cerr << "removing linearly dependent columns" << endl;

    // Remove linearly dependent columns.  Returns a vector saying in which
    // entry of the new matrix the current column really is
    vector<int> new_loc = remove_dependent(outputs);
    

    Link_Function blend_link_function
        = (target == AUC ? LOGIT : LINEAR);

    cerr << "training blender" << endl;

    distribution<BlendFloat> trained
        = run_irls(correct, outputs, w, blend_link_function);

    distribution<float> parameters(nv);
    for (unsigned v = 0;  v < nv;  ++v)
        if (new_loc[v] != -1) parameters[v] = trained[new_loc[v]];

    cerr << "blend coefficients: " << parameters << endl;

    blend_coefficients = parameters;
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
                   const distribution<double> & target_singular,
                   const Target_Stats & target_stats) const
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
    //result.insert(result.end(),
    //              model_outputs.begin(), model_outputs.end());
    result.push_back(model_prediction_10 - real_prediction);
    result.push_back(fabs(result.back()));
    result.push_back(model_prediction_50 - real_prediction);
    result.push_back(fabs(result.back()));

    // 5.  Target Mean output
    // 6.  Target standard deviation of output
    // 7.  Target min output
    // 8.  Target max output
    // 9.  Distance from an integer
    // ...

    result.push_back(target_stats.mean);
    result.push_back(target_stats.std);
    result.push_back(target_stats.min);
    result.push_back(target_stats.max);

    result.push_back((real_prediction - target_stats.mean) / target_stats.std);
    result.push_back((real_prediction - target_stats.min) / target_stats.std);
    result.push_back((real_prediction - target_stats.max) / target_stats.std);

    result.push_back(std::min(fabs(real_prediction - ceil(real_prediction)),
                              fabs(real_prediction - floor(real_prediction))));

    result.push_back(target_stats.max - target_stats.min);
    result.push_back((target_stats.max - target_stats.mean) / target_stats.std);
    result.push_back((target_stats.mean - target_stats.min) / target_stats.std);

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

    Target_Stats target_stats(models.begin(), models.end());

    for (unsigned i = 0;  i < models.size();  ++i) {
        // Skip untrained models
        if (model_coefficients[i].empty()) continue;

        // What would we have predicted for this model?

        distribution<float> model_features
            = get_model_features(i, models, target_singular, target_stats);

        // Perform linear regression (in prediction mode)
        float output = model_features.dotprod(model_coefficients[i]);

        // Link function to change into a probability
        float prob = apply_link_inverse(output, link_function);

        result[i] = prob;
    }

    return result;
}

distribution<float>
Gated_Blender::
get_blend_features(const distribution<float> & model_outputs,
                   const distribution<float> & model_conf) const
{
    distribution<float> result;

    result.push_back(model_outputs.min());
    result.push_back(model_outputs.max());
    result.push_back(model_outputs.total() / (model_conf != 0.0).count());

    distribution<float> weighted = model_outputs * model_conf;
    result.push_back(weighted.min());
    result.push_back(weighted.max());
    result.push_back(weighted.total() / (model_conf != 0.0).count());
    result.push_back(weighted.total() / model_conf.total());

    distribution<float> dense_model, dense_conf;

    for (unsigned i = 0;  i < model_outputs.size();  ++i) {
        if (model_coefficients[i].empty()) continue;
        result.push_back(model_outputs[i]);
        result.push_back(model_conf[i]);
        result.push_back(weighted[i]);
        dense_model.push_back(model_outputs[i]);
        dense_conf.push_back(model_conf[i]);
    }

    result.push_back(dense_conf.min());
    result.push_back(dense_conf.max());
    result.push_back(dense_conf.total() / dense_conf.size());

    distribution<bool> ismax = dense_conf == dense_conf.max();

    result.push_back((dense_model * ismax).total() / (ismax.count()));
    result.push_back((dense_model * dense_conf * ismax).total()
                     / (ismax.count()));

    result.push_back(model_outputs.total() / model_outputs.size());
    result.push_back(dense_model.total() / dense_model.size());

    return result;
}

float
Gated_Blender::
predict(const ML::distribution<float> & models) const
{
    bool debug = debug_predict;

    auto_ptr<Guard> guard;

    if (debug) {
        static Lock lock;
        guard.reset(new Guard(lock));
    }

    distribution<float> conf = this->conf(models);
    
    distribution<float> model_preds(models.size());
    for (unsigned i = 0;  i < models.size();  ++i)
        model_preds[i] = (models[i] > 3.0 ? 1.0 : -1.0);

    //float result = model_preds.dotprod(conf) / conf.total();

    //float result = models.dotprod(conf) / conf.total();

    //float result = conf.total() * 0.1 * 4.0 + 1.0;
   
    for (unsigned i = 0;  i < models.size() && debug;  ++i) {
        if (conf[i] == 0.0) continue;
        cerr << "model " << i << ": pred " << models[i] << " conf "
             << conf[i] << endl;
    }

    distribution<float> blend_features
        = get_blend_features(models, conf);

    Link_Function blend_link_function
        = (target == AUC ? LOGIT : LINEAR);

    float result
        = apply_link_inverse(blend_features.dotprod(blend_coefficients),
                             blend_link_function);

    if (debug) cerr << "result = " << result << " correct = "
                    << correct_prediction << endl;

    return result;
}
