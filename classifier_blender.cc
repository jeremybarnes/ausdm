/* classifier_blender.cc
   Jeremy Barnes, 2 November 2009
   Copyright (c) 2009 Jeremy Barnes.  All rights reserved.

   Blender for multiple regression models.
*/

#include "classifier_blender.h"
#include "utils.h"
#include "jml/utils/worker_task.h"
#include "jml/utils/guard.h"
#include "jml/boosting/classifier_generator.h"
#include "decomposition.h"
#include "jml/stats/distribution_ops.h"

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
    use_recomp_features = true;
    config.find(use_decomposition_features, "use_decomposition_features");
    config.find(use_extra_features, "use_extra_features");
    config.find(use_recomp_features, "use_recomp_features");

    use_regression = true;
    config.find(use_regression, "use_regression");

    this->random_seed = random_seed;
    this->target = target;

    debug_predict = false;
    config.find(debug_predict, "debug_predict");

    flatten_range = false;
    config.find(flatten_range, "flatten_range");
}

void
Classifier_Blender::
generate_training_data(Training_Data & cls_training_data,
                       const Data & train) const
{
    for (unsigned i = 0;  i < train.nx();  ++i) {

        float correct;
        if (target == AUC)
            correct = train.targets[i] > 0.0;
        else
            correct = train.targets[i];

        correct_prediction = correct;

        //cerr << "correct_prediction = " << correct_prediction << endl;

        const distribution<float> & model_outputs
            = train.examples[i]->models;

        const Target_Stats & target_stats
            = train.examples[i]->stats;
        
        const distribution<float> & target_singular
            = train.examples[i]->decomposed;
        
        distribution<float> features
            = get_features(model_outputs, target_singular, target_stats);
        
        if (features.size() != nv)
            throw Exception("nv is wrong");
        
        if (target == AUC)
            features.push_back(correct);
        else if (use_regression && !flatten_range)
            features.push_back(correct);
        else if (use_regression && flatten_range)
            features.push_back((correct + 1.0) * 0.50);
        else {
            features.push_back(correct > -1.0);
            features.push_back(correct > -0.5);
            features.push_back(correct > 0.0);
            features.push_back(correct > 0.5);
            features.push_back(correct == -1.0);
            features.push_back(correct == -0.5);
            features.push_back(correct ==  0.0);
            features.push_back(correct ==  0.5);
        }
        
        boost::shared_ptr<Mutable_Feature_Set> fset
            = fs->encode(features);
        
        //if (dump_training_features != "")
        //    training_feature_file << fs->print(*fset) << endl;

        cls_training_data.add_example(fset);
    }
}

void
Classifier_Blender::
init(const Data & training_data,
     const ML::distribution<float> & example_weights)
{
    Data train = training_data, test;
    distribution<float> training_weights = example_weights, testing_weights;

    train.hold_out(test, 0.2, training_weights, testing_weights);

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

    int nlabels = 1;
    
    if (target == AUC) {
        fs->add_feature("LABEL",
                        Feature_Info(BOOLEAN,
                                     false /* optional */,
                                     true /* biased */));
    }
    else {
        if (use_regression) {
            fs->add_feature("LABEL",
                            Feature_Info(REAL,
                                         false /* optional */,
                                         true /* biased */));
        }
        else {
            nlabels = 8;
            for (unsigned i = 0;  i < nlabels;  ++i) {
                fs->add_feature(format("LABEL%d", i),
                                Feature_Info(BOOLEAN, false, true));
            }
        }
    }
    
    //if (dump_training_features != "")
    //    training_feature_file << fs->print() << endl;
    
    //if (dump_predict_features != "")
    //    predict_feature_file << fs->print() << endl;

    Training_Data cls_training_data(fs);
    Training_Data cls_testing_data(fs);

    generate_training_data(cls_training_data, train);
    generate_training_data(cls_testing_data, test);

    Configuration config;
    config.load(trainer_config_file);

    if (target == AUC || use_regression) {
        boost::shared_ptr<Classifier_Generator> trainer
            = get_trainer(trainer_name, config);
        
        trainer->init(fs, Feature(nv));

        Thread_Context context;

        classifier = trainer->generate(context, cls_training_data,
                                       training_weights,
                                       cls_training_data.all_features());
    }
    else {
        vector<Feature> all_features
            = cls_training_data.all_features();

        // Remove the 4 label features
        for (unsigned i = 0;  i < nlabels;  ++i)
            all_features.pop_back();

        classifiers.resize(nlabels);
        probabilizers.resize(nlabels);

        for (unsigned i = 0;  i < nlabels;  ++i) {
            cerr << "training model " << i << endl;
            
            boost::shared_ptr<Classifier_Generator> trainer
                = get_trainer(trainer_name, config);
        
            trainer->init(fs, Feature(nv + i));
            
            Thread_Context context;
            
            classifiers[i] = trainer->generate(context, cls_training_data,
                                               training_weights,
                                               all_features);
            Optimization_Info opt
                = classifiers[i]->optimize(cls_training_data.all_features());

            probabilizers[i].train(cls_testing_data,
                                   *classifiers[i],
                                   opt, 2, "logit");
        }

        int nxt = cls_testing_data.example_count();

        vector<distribution<float> > outputs(cls_testing_data.example_count());
        int nv = 0;
        for (unsigned i = 0;  i < cls_testing_data.example_count();  ++i) {
            distribution<float> sub_results
                = predict_rmse_binary_features(cls_testing_data[i]);
            outputs[i] = sub_results;
            if (i == 0) nv = outputs[i].size();
        }

        Thread_Context context;

        int num_iter = 20;

        for (unsigned l = 0;  l < num_iter;  ++l) {

            vector<int> examples;
            for (unsigned i = 0;  i < nxt;  ++i)
                if (context.random01() < 0.80)
                    examples.push_back(i);
        
            int nxt2 = examples.size();

            boost::multi_array<double, 2> values(boost::extents[nxt2][nv]);
            distribution<double> correct(nxt2);
            
            for (unsigned i = 0;  i < nxt2;  ++i) {
                int x = examples[i];

                correct[i] = test.targets[x];
                for (unsigned j = 0;  j < nv;  ++j) {
                    values[i][j] = outputs[x][j];
                }
            }
            
            Ridge_Regressor regressor(1e-5);
            
            if (l == 0)
                params = regressor.calc(values, correct) / num_iter;
            else params += regressor.calc(values, correct) / num_iter;
        }

        cerr << "params = " << params << endl;
    }
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

        for (unsigned i = 0;
             i < recomposition_orders.size() && use_recomp_features;
             ++i) {
            string s2 = format("%s_recomp_error_%d", s.c_str(),
                               recomposition_orders[i]);
            result->add_feature(s2, REAL);
            result->add_feature(s2 + "_abs", REAL);
            result->add_feature(s2 + "_sqr", REAL);
        }
    }

    for (unsigned i = 0;
         i < recomposition_orders.size() && use_recomp_features;  ++i) {
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

    for (unsigned i = 0;  i < recomposition_orders.size() && use_recomp_features;  ++i) {
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
        
        for (unsigned r = 0;  r < recomposition_orders.size() && use_recomp_features;  ++r) {
            const distribution<float> & reconst = recompositions[r];
            result.push_back(reconst[i] - model_outputs[i]);
            result.push_back(abs(reconst[i] - model_outputs[i]));
            result.push_back(pow(reconst[i] - model_outputs[i], 2));
        }
    }

    for (unsigned i = 0;  i < recomposition_orders.size() && use_recomp_features;  ++i) {
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

distribution<float>
Classifier_Blender::
predict_rmse_binary_features(const ML::Feature_Set & fset) const
{
    distribution<float> result;

    distribution<float> sub_results(4);
    for (unsigned i = 0;  i < 4;  ++i)
        sub_results[i] = classifiers[i]->predict(1, fset);
    
    result.extend(sub_results);

    distribution<float> prob_results(4);
    for (unsigned i = 0;  i < 4;  ++i)
        prob_results[i] = probabilizers[i].apply(classifiers[i]->predict(fset))[1];
    
    result.extend(prob_results);

    distribution<float> class_probs(5);
    class_probs[0] = 1.0 - sub_results[0];
    class_probs[1] = sub_results[0] - sub_results[1];
    class_probs[2] = sub_results[1] - sub_results[2];
    class_probs[3] = sub_results[2] - sub_results[3];
    class_probs[4] = sub_results[3];
    
    result.extend(class_probs);

    for (unsigned i = 0;  i < 5;  ++i)
        if (class_probs[i] < 0.0)
            class_probs[i] = 0.0;
    
    distribution<float> class_probs_norm = class_probs;
    class_probs_norm.normalize();
    
    result.extend(class_probs_norm);

    result.extend(class_probs == class_probs.max());

    distribution<float> sub_results2(5);
    distribution<float> prob_results2(5);
    sub_results2[0] = classifiers[4]->predict(1, fset);
    sub_results2[1] = classifiers[5]->predict(1, fset);
    sub_results2[2] = classifiers[6]->predict(1, fset);
    sub_results2[3] = classifiers[7]->predict(1, fset);
    sub_results2[4] = sub_results[3];

    result.extend(sub_results2);

    prob_results2[0] = probabilizers[4].apply(classifiers[4]->predict(fset))[0];
    prob_results2[1] = probabilizers[5].apply(classifiers[5]->predict(fset))[0];
    prob_results2[2] = probabilizers[6].apply(classifiers[6]->predict(fset))[0];
    prob_results2[3] = probabilizers[7].apply(classifiers[7]->predict(fset))[0];
    prob_results2[4] = prob_results[3];
    
    result.extend(prob_results2);

    distribution<float> class_probs2 = prob_results2;
    class_probs2.normalize();

    result.extend(class_probs2);
    

    distribution<float> class_values(5);
    class_values[0] = -1.0;
    class_values[1] = -0.5;
    class_values[2] = 0.0;
    class_values[3] = 0.5;
    class_values[4] = 1.0;
    
    result.push_back(class_probs_norm.dotprod(class_values));
    result.push_back(class_values.dotprod(class_probs == class_probs.max()));

    result.push_back(class_probs2.dotprod(class_values));
    result.push_back(class_values.dotprod(class_values == class_values.max()));

    float highest = 0.0, second_highest = 0.0;
    int nhighest = -1, nsecond_highest = -1;
    for (unsigned i = 0;  i < 5;  ++i) {
        if (class_probs_norm[i] > highest) {
            second_highest = highest;
            nsecond_highest = nhighest;
            highest = class_probs_norm[i];
            nhighest = i;
        }
        else if (class_probs_norm[i] > second_highest) {
            second_highest = class_probs_norm[i];
            nsecond_highest = i;
        }
    }
    
    distribution<float> margins(5);
    for (unsigned i = 0;  i < 5;  ++i)
        margins[i] = (i == highest
                      ? highest - second_highest
                      : class_probs_norm[i] - highest);

    result.extend(margins);

    result.push_back(1.0); // bias

    return result;
}

float
Classifier_Blender::
predict(const ML::distribution<float> & models) const
{
    distribution<float> features = get_features(models);

    if (target == AUC || use_regression)
        features.push_back(correct_prediction);
    else {
        features.push_back(correct_prediction > -1.0);
        features.push_back(correct_prediction > -0.5);
        features.push_back(correct_prediction >  0.0);
        features.push_back(correct_prediction >  0.5);
        features.push_back(correct_prediction == -1.0);
        features.push_back(correct_prediction == -0.5);
        features.push_back(correct_prediction ==  0.0);
        features.push_back(correct_prediction ==  0.5);
    }

    boost::shared_ptr<Mutable_Feature_Set> feature_set
        = fs->encode(features);

    float result;
    if (target == AUC) 
        result = classifier->predict(1, *feature_set);
    else if (use_regression && !flatten_range)
        result = classifier->predict(0, *feature_set);
    else if (use_regression && flatten_range)
        result = classifier->predict(0, *feature_set) * 2.0 - 1.0;
    else {
        distribution<float> sub_results
            = predict_rmse_binary_features(*feature_set);

        result = sub_results.dotprod(params);

        if (debug_predict) {
            static Lock lock;
            Guard guard(lock);
            cerr << endl << "---------------------------" << endl;
            cerr << "correct: " << correct_prediction << endl;
            cerr << "sub_results: " << sub_results << endl;
            cerr << "result: " << result << " correct: " << correct_prediction
                 << endl;
        }

        if (result < -1) result = -1;
        if (result > 1) result = 1;
    }

    return result;
}

std::string
Classifier_Blender::
explain(const ML::distribution<float> & models) const
{
    if (target == RMSE && !use_regression) return "";
    
    distribution<float> features = get_features(models);

    features.push_back(correct_prediction);

    boost::shared_ptr<Mutable_Feature_Set> feature_set
        = fs->encode(features);

    ML::Explanation explanation
        = classifier->explain(*feature_set, target == AUC);
    
    return explanation.print();
}

