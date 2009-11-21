/* classifier_blender.cc
   Jeremy Barnes, 2 November 2009
   Copyright (c) 2009 Jeremy Barnes.  All rights reserved.

   Blender for multiple regression models.
*/

#include "classifier_blender.h"
#include "utils.h"
#include "boosting/worker_task.h"
#include "utils/guard.h"
#include "boosting/classifier_generator.h"
#include "decomposition.h"

#include <boost/bind.hpp>
#include <boost/progress.hpp>

using namespace std;
using namespace ML;


/*****************************************************************************/
/* CLASSIFIER_BLENDER                                                        */
/*****************************************************************************/

Classifier_Blender::
Classifier_Blender()
    : decomposition(0)
{
}

Classifier_Blender::
~Classifier_Blender()
{
}

void
Classifier_Blender::
configure(const ML::Configuration & config_,
          const std::string & name,
          int random_seed,
          Target target)
{
    Configuration config(config_, name, Configuration::PREFIX_APPEND);

    trainer_config_file = "classifier-config.txt";
    config.find(trainer_config_file, "config_file");

    trainer_name = (target == AUC ? "auc" : "rmse");
    config.find(trainer_name, "trainer_name");

    use_decomposition_features = true;
    use_extra_features = true;
    config.find(use_decomposition_features, "use_decomposition_features");
    config.find(use_extra_features, "use_extra_features");

    this->random_seed = random_seed;
    this->target = target;
}

void
Classifier_Blender::
init(const Data & training_data,
     const ML::distribution<float> & example_weights)
{
    Data train = training_data, test;
    distribution<float> training_weights = example_weights, testing_weights;

    this->decomposition = training_data.decomposition;
    this->model_stats = training_data.models;
    this->model_names = training_data.model_names;
    this->nm = model_names.size();

    if (decomposition) {
        recomposition_orders = decomposition->recomposition_orders();
        ndecomposed = decomposition->size();
    }
    else ndecomposed = 0;

    nv = get_features(training_data.examples[0]->models,
                      training_data.examples[0]->decomposed,
                      training_data.examples[0]->stats).size();


    fs = feature_space();
    
    if (fs->features().size() != nv) {
        cerr << "fs: " << fs->features().size() << endl;
        cerr << "nv: " << nv << endl;
        throw Exception("blend feature space has wrong number of features");
    }
    
    fs->add_feature("LABEL",
                    Feature_Info(target == AUC ? BOOLEAN : REAL,
                                 false /* optional */,
                                 true /* biased */));
    
    //if (dump_training_features != "")
    //    training_feature_file << fs->print() << endl;
    
    //if (dump_predict_features != "")
    //    predict_feature_file << fs->print() << endl;

    Training_Data cls_training_data(fs);

    for (unsigned i = 0;  i < training_data.nx();  ++i) {

        float correct;
        if (target == AUC)
            correct = training_data.targets[i] > 0.0;
        else
            correct = training_data.targets[i];

        correct_prediction = correct;

        //cerr << "correct_prediction = " << correct_prediction << endl;

        const distribution<float> & model_outputs
            = training_data.examples[i]->models;

        const Target_Stats & target_stats
            = training_data.examples[i]->stats;
        
        const distribution<float> & target_singular
            = training_data.examples[i]->decomposed;
        
        distribution<float> features
            = get_features(model_outputs, target_singular, target_stats);
        
        if (features.size() != nv)
            throw Exception("nv is wrong");
        
        features.push_back(correct);
        
        boost::shared_ptr<Mutable_Feature_Set> fset
            = fs->encode(features);
        
        //if (dump_training_features != "")
        //    training_feature_file << fs->print(*fset) << endl;

        cls_training_data.add_example(fset);
    }

    Configuration config;
    config.load(trainer_config_file);

    boost::shared_ptr<Classifier_Generator> trainer
        = get_trainer(trainer_name, config);

    trainer->init(fs, Feature(nv));

    Thread_Context context;

    classifier = trainer->generate(context, cls_training_data,
                                   training_weights,
                                   cls_training_data.all_features());

    // TODO: probabilizer for AUC
}

boost::shared_ptr<Dense_Feature_Space>
Classifier_Blender::
feature_space() const
{
    boost::shared_ptr<Dense_Feature_Space> result
        (new Dense_Feature_Space());

    for (unsigned i = 0;  i != nm;  ++i)
        result->add_feature(model_names[i], REAL);

    for (unsigned i = 0;  i  !=  ndecomposed && use_decomposition_features;  ++i)
        result->add_feature(format("decomp%03d", i), REAL);
    
    if (!use_extra_features) return result;
    
    result->add_feature("min_model", REAL);
    result->add_feature("max_model", REAL);
         
    
    for (unsigned i = 0;  i < nm;  ++i) {
        if (model_stats[i].rank >= 10)
            continue;
        string s = model_names[i];
        result->add_feature(s + "_output", REAL);
        result->add_feature(s + "_dev_from_mean", REAL);
        result->add_feature(s + "_diff_from_int", REAL);

        for (unsigned i = 0;  i < recomposition_orders.size();  ++i) {
            string s2 = format("%s_recomp_error_%d", s.c_str(),
                               recomposition_orders[i]);
            result->add_feature(s2, REAL);
            result->add_feature(s2 + "_abs", REAL);
            result->add_feature(s2 + "_sqr", REAL);
        }
    }

    for (unsigned i = 0;  i < recomposition_orders.size();  ++i) {
        string s = format("recomp_error_%d", recomposition_orders[i]);
        result->add_feature(s + "_rmse", REAL);
    }
    
    result->add_feature("avg_model_chosen", REAL);

    result->add_feature("models_mean", REAL);
    result->add_feature("models_std", REAL);
    result->add_feature("models_min", REAL);
    result->add_feature("models_max", REAL);
    
    result->add_feature("models_range", REAL);
    result->add_feature("models_range_dev_high", REAL);
    result->add_feature("models_range_dev_low", REAL);

    result->add_feature("diff_mean_10_all", REAL);
    result->add_feature("abs_diff_mean_10_all", REAL);
    
    return result;
}

distribution<float>
Classifier_Blender::
get_features(const ML::distribution<float> & models) const
{
    distribution<float> decomposed;
    if (decomposition)
        decomposed = decomposition->decompose(models);
    Target_Stats stats(models.begin(), models.end());

    return get_features(models, decomposed, stats);
}

distribution<float>
Classifier_Blender::
get_features(const ML::distribution<float> & model_outputs,
             const ML::distribution<float> & decomp,
             const Target_Stats & stats) const
{
    distribution<float> result = model_outputs;

    if (use_decomposition_features) result.extend(decomp);

    if (!use_extra_features) return result;

    result.push_back(model_outputs.min());
    result.push_back(model_outputs.max());

    vector<distribution<float> > recompositions;

    for (unsigned i = 0;  i < recomposition_orders.size();  ++i) {
        int nr = recomposition_orders[i];
        distribution<float> reconst;
        if (!decomposition) reconst = model_outputs;
        else reconst
                 = decomposition
                 ->recompose(model_outputs, decomp, nr);
        recompositions.push_back(reconst);
    }

    distribution<float> dense_model;

    for (unsigned i = 0;  i < model_outputs.size();  ++i) {
        if (model_stats[i].rank >= 10) continue;
        result.push_back(model_outputs[i]);
        dense_model.push_back(model_outputs[i]);

        float real_prediction = model_outputs[i];

        result.push_back((real_prediction - stats.mean)
                         / stats.std);
        result.push_back
            (std::min(fabs(real_prediction - ceil(real_prediction)),
                      fabs(real_prediction - floor(real_prediction))));
        
        for (unsigned r = 0;  r < recomposition_orders.size();  ++r) {
            const distribution<float> & reconst = recompositions[r];
            result.push_back(reconst[i] - model_outputs[i]);
            result.push_back(abs(reconst[i] - model_outputs[i]));
            result.push_back(pow(reconst[i] - model_outputs[i], 2));
        }
    }

    for (unsigned i = 0;  i < recomposition_orders.size();  ++i) {
        const distribution<float> & reconst = recompositions[i];
        result.push_back((reconst - model_outputs).two_norm());
    }
    
    float avg_model_chosen = dense_model.mean();

    result.push_back(avg_model_chosen);

    result.push_back(stats.mean);
    result.push_back(stats.std);
    result.push_back(stats.min);
    result.push_back(stats.max);

    result.push_back(stats.max - stats.min);
    result.push_back((stats.max - stats.mean) / stats.std);
    result.push_back((stats.mean - stats.min) / stats.std);


    result.push_back(stats.mean - avg_model_chosen);
    result.push_back(abs(stats.mean - avg_model_chosen));

    return result;
}

float
Classifier_Blender::
predict(const ML::distribution<float> & models) const
{
    distribution<float> features = get_features(models);
    features.push_back(correct_prediction);

    boost::shared_ptr<Mutable_Feature_Set> feature_set
        = fs->encode(features);

    float result;
    if (target == AUC)
        result = classifier->predict(1, *feature_set);
    else result = classifier->predict(0, *feature_set);

    return result;
}

std::string
Classifier_Blender::
explain(const ML::distribution<float> & models) const
{
    distribution<float> features = get_features(models);

    features.push_back(correct_prediction);

    boost::shared_ptr<Mutable_Feature_Set> feature_set
        = fs->encode(features);

    ML::Explanation explanation
        = classifier->explain(*feature_set, target == AUC);
    
    return explanation.print();
}

