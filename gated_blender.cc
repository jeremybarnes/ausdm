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
#include "algebra/least_squares.h"
#include "stats/distribution_ops.h"
#include "utils/filter_streams.h"
#include "boosting/classifier_generator.h"
#include "decomposition.h"
#include "utils.h"


using namespace ML;
using namespace ML::Stats;
using ML::Stats::sqr;
using namespace std;


/*****************************************************************************/
/* GATED_BLENDER                                                             */
/*****************************************************************************/

Gated_Blender::Gated_Blender()
    : link_function(LOGIT)
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

    config.find(dump_training_features, "dump_training_features");
    config.find(dump_predict_features, "dump_predict_features");

    blender_trainer_config_file = "classifier-config.txt";
    config.find(blender_trainer_config_file, "blender.config_file");

    blender_trainer_name = (target == AUC ? "blender_auc" : "blender_rmse");
    config.find(blender_trainer_name, "blender.trainer_name");

    this->target = target;
}

void
Gated_Blender::
train_conf(int model, const Data & training_data,
           const ML::distribution<float> & example_weights)
{
    // Generate a matrix with the predictions
    int nx = training_data.nx();

    if (nx == 0)
        throw Exception("can't train with no examples");

    int nv = get_conf_features
        (model,
         training_data.examples[0],
         training_data.singular_targets[0],
         Target_Stats())
        .size();


    typedef double Float;

    // Assemble the labels
    distribution<Float> correct(nx);
    boost::multi_array<Float, 2> outputs(boost::extents[nv][nx]);
    distribution<Float> w(example_weights.begin(), example_weights.end());

    for (unsigned i = 0;  i < training_data.nx();  ++i) {

        if (target == AUC) {
            // For now, we will try to predict if the margin > 0.  This only
            // works for AUC; for RMSE we will need something different.
            // Eventually, we might want to predict the margin directly or
            // take a threshold for the margin, eg 0.5
            float pred = training_data.models[model][i];
            float margin = pred * training_data.targets[i];
            
            correct[i] = (margin >= 0.2);
            //correct[i] = training_data.targets[i] > 0.0;
            //correct[i] = (margin * 0.5) + 0.5;
        }
        else {
            // Try to predict the probability that it's within 0.5 either side
            // of the correct answer.  With our modified scale, this really
            // means predicting if it's within 0.25.
            correct[i]
                = abs(training_data.models[model][i]
                      - training_data.targets[i]) <= 0.25;
        }

        const distribution<float> & model_outputs
            = training_data.examples[i];
        
#if 0
        cerr << "training_data.singular_targets.size() = "
             << training_data.singular_targets.size() << endl;
        cerr << "i = " << i << endl;
        cerr << "training_data.nx() = " << training_data.nx() << endl;
#endif

        const distribution<float> & target_singular
            = training_data.singular_targets[i];

        distribution<float> features
            = get_conf_features(model, model_outputs, target_singular,
                                training_data.target_stats[i]);

        //cerr << "conf features: " << features << endl;

        if (features.size() != nv)
            throw Exception("nv is wrong");

        for (unsigned j = 0;  j < nv;  ++j)
            outputs[j][i] = features[j];
    }

    distribution<Float> parameters(nv);
    int num_good = 0;

    Thread_Context context;

    int n_irls = 5;
    for (unsigned i = 0;  i < n_irls;  ++i) {
        
        float p_in = min(1.0, 2.0 / n_irls);

        vector<int> examples;
        for (unsigned x = 0;  x < nx;  ++x) {
            if (context.random01() >= p_in) continue;
            examples.push_back(x);
        }

        int nx2 = examples.size();

        distribution<Float> correct2(nx2);
        boost::multi_array<Float, 2> outputs2(boost::extents[nv][nx2]);
        distribution<Float> w2(nx2);

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

        if (trained_params.two_norm() > 200.0) {
            cerr << format("trained_params.two_norm() = %f\n",
                           trained_params.two_norm());
            continue;
        }
        
        parameters += trained_params;
        ++num_good;
    }

    parameters /= num_good;

    //cerr << "parameters for model " << model << ": " << parameters << endl;

    Model_Output before, after;
    before.resize(training_data.nx());
    after.resize(training_data.nx());

    // Test the original model and the weighted version for AUC
    for (unsigned i = 0;  i < training_data.nx();  ++i) {

        distribution<float> features(nv);
        for (unsigned j = 0;  j < nv;  ++j)
            features[j] = outputs[j][i];
        
        float result = apply_link_inverse(features.dotprod(parameters),
                                          link_function);

        before[i] = training_data.models[model][i];
        after[i] = result;
    }

    //cerr << "before = " << before << endl;

    float auc_before1 = before.calc_score(training_data.targets, target);
    float auc_after1  = after.calc_score(training_data.targets, target);
    float auc_before2 = before.calc_auc(correct.cast<float>() * 2.0f - 1.0f);
    float auc_after2  = after.calc_auc(correct.cast<float>() * 2.0f - 1.0f);

    static Lock lock;
    Guard guard(lock);
    
    cerr << "model " << model
         << ": before " << auc_before1 << "/" << auc_before2
         << " after " << auc_after1 << "/" << auc_after2 << endl;

    if (auc_after2 < 0.01) {
        cerr << "error on auc_after2" << endl;
        //cerr << "new_loc = " << new_loc << endl;
        cerr << "parameters = " << parameters << endl;
        //cerr << "trained = " << trained << endl;
        //cerr << "svalues_reduced = " << svalues_reduced << endl;
        
        cerr << "after = " << distribution<float>(after.begin(),
                                                  after.begin() + 100)
             << endl;

        for (unsigned i = 0;  i < 10;  ++i) {
            cerr << "example " << i << ":" << endl;

            distribution<float> features(nv);
            for (unsigned j = 0;  j < nv;  ++j)
                features[j] = outputs[j][i];

            //cerr << "  features = " << features << endl;
            //cerr << "  mult = " << features * parameters << endl;
            cerr << "  dotprod = " << features.dotprod(parameters)
                 << endl;

            float result = apply_link_inverse(features.dotprod(parameters),
                                              link_function);

#if 0            
            distribution<float> reduced(nkeep);
            for (unsigned j = 0;  j < nv;  ++j)
                if (new_loc[j] != -1)
                    reduced[new_loc[j]] = outputs[j][i];
            
            cerr << "  dotprod2 = " << reduced.dotprod(trained) << endl;
            float result2 = apply_link_inverse(reduced.dotprod(trained),
                                               link_function);
#endif
            cerr << "  result  = " << result << endl;
            //cerr << "  result2 = " << result2 << endl;

        }

        throw Exception("error on auc_after2");
    }

    model_coefficients[model] = parameters;
}

struct Generate_Blend_Data_Job {
};

void
Gated_Blender::
init(const Data & training_data_in,
     const ML::distribution<float> & example_weights)
{
    if (dump_predict_features != "")
        predict_feature_file.open(dump_predict_features);

    if (dump_training_features != "")
        training_feature_file.open(dump_training_features);

    // Perform the decomposition, one time, on all training+testing
    // data (we don't look at the labels, so this is OK).

    string targ_type_uc;
    if (target == AUC) targ_type_uc = "AUC";
    else if (target == RMSE) targ_type_uc = "RMSE";
    else throw Exception("unknown target type");

    this->data = &training_data_in;

    if (data->decomposition)
        recomposition_sizes = data->decomposition->recomposition_orders();
    
    //decompose_training_data.load("download/S_"
    //                             + targ_type_uc + "_Train.csv", target);

    //decompose_training_data.calc_scores();

    //decompose_training_data.load("download/S_"
    //                             + targ_type_uc + "_Score.csv", target,
    //                             false);

    Data conf_training_data = training_data_in;
    Data blend_training_data;

    distribution<float> conf_example_weights = example_weights,
        blend_example_weights;

    // One third for the decomposition, one third for the confidence, one third
    // for the blender

    //decompose_training_data.hold_out(conf_training_data, 2.0 / 3.0);
    conf_training_data.hold_out(blend_training_data, 0.5,
                                conf_example_weights, blend_example_weights);

    //cerr << "training decomposition" << endl;
    //decompose_training_data.decompose();
    //conf_training_data.apply_decomposition(decompose_training_data);
    //blend_training_data.apply_decomposition(decompose_training_data);

    //decompose_training_data.stats();
    conf_training_data.stats();
    blend_training_data.stats();

    //decompose_training_data.calc_scores();
    conf_training_data.calc_scores();
    blend_training_data.calc_scores();

    int nm = conf_training_data.nm();

    // Now to train.  For each of the models, we go through the training
    // data and create a data file; we then do an IRLS on the model.

    model_coefficients.resize(nm);


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

        for (unsigned i = 0;  i < nm;  ++i) {
            if (training_data_in.models[i].rank >= num_models_to_train)
                continue;

            worker.add(boost::bind(&Gated_Blender::train_conf,
                                   this, i, boost::cref(conf_training_data),
                                   boost::cref(conf_example_weights)),
                       "train model job",
                       group);
        }
    }

    // Add this thread to the thread pool until we're ready
    worker.run_until_finished(group);


    int nx = blend_training_data.nx();

    int nv = get_blend_features
        (distribution<float>(nm),
         distribution<float>(nm),
         blend_training_data.singular_targets[0],
         Target_Stats())
        .size();


    cerr << "generating blend data" << endl;

    typedef double BlendFloat;

    // Assemble the labels
    distribution<BlendFloat> correct(nx);
    //boost::multi_array<BlendFloat, 2> outputs(boost::extents[nv][nx]);
    distribution<BlendFloat> w(blend_example_weights.begin(),
                               blend_example_weights.end());
    w.normalize();

    boost::shared_ptr<ML::Dense_Feature_Space> fs = blend_feature_space();

    if (fs->features().size() != nv) {
        cerr << "fs: " << fs->features().size() << endl;
        cerr << "nv: " << nv << endl;
        throw Exception("blend feature space has wrong number of features");
    }

    fs->add_feature("LABEL", (target == AUC ? BOOLEAN : REAL));

    blender_fs = fs;

    if (dump_training_features != "")
        training_feature_file << fs->print() << endl;

    if (dump_predict_features != "")
        predict_feature_file << fs->print() << endl;

    Training_Data training_data(fs);

    for (unsigned i = 0;  i < blend_training_data.nx();  ++i) {

        if (target == AUC)
            correct[i] = blend_training_data.targets[i] > 0.0;
        else
            correct[i] = blend_training_data.targets[i];

        correct_prediction = correct[i];

        //cerr << "correct_prediction = " << correct_prediction << endl;

        const distribution<float> & model_outputs
            = blend_training_data.examples[i];

        const Target_Stats & target_stats
            = blend_training_data.target_stats[i];
        
        const distribution<float> & target_singular
            = blend_training_data.singular_targets[i];

        distribution<float> conf
            = this->conf(model_outputs, target_singular, target_stats);

        distribution<float> features
            = get_blend_features(model_outputs, conf, target_singular,
                                 target_stats);

        if (features.size() != nv)
            throw Exception("nv is wrong");

        //for (unsigned j = 0;  j < nv;  ++j)
        //    outputs[j][i] = features[j];

        features.push_back(correct[i]);
        
        boost::shared_ptr<Mutable_Feature_Set> fset
            = fs->encode(features);

        if (dump_training_features != "")
            training_feature_file << fs->print(*fset) << endl;

        training_data.add_example(fset);

    }

#if 0
    cerr << "training blender" << endl;

    Link_Function blend_link_function
        = (target == AUC ? LOGIT : LINEAR);

    distribution<BlendFloat> parameters
        = perform_irls(correct, outputs, w, blend_link_function);
    
    cerr << "blend coefficients: " << parameters << endl;

    blend_coefficients = parameters;
#endif

    Configuration config;
    config.load(blender_trainer_config_file);

    boost::shared_ptr<Classifier_Generator> trainer
        = get_trainer(blender_trainer_name, config);

    trainer->init(fs, Feature(nv));

    Thread_Context context;

    blender = trainer->generate(context, training_data, blend_example_weights,
                                training_data.all_features());
}

boost::shared_ptr<Dense_Feature_Space>
Gated_Blender::
conf_feature_space() const
{
    boost::shared_ptr<Dense_Feature_Space> result
        (new Dense_Feature_Space());

    result->add_feature("bias", REAL);

    result->add_feature("model_pred", REAL);

    for (unsigned i = 0;  i < data->singular_targets[0].size();  ++i)
        result->add_feature(format("pc%03d", i), REAL);

    for (unsigned i = 0;  i < recomposition_sizes.size();  ++i) {
        string s = format("error_%d", recomposition_sizes[i]);
        result->add_feature(s, REAL);
        result->add_feature(s + "_abs", REAL);
        result->add_feature(s + "_sqr", REAL);
        result->add_feature(s + "_rmse", REAL);
    }

    result->add_feature("models_mean", REAL);
    result->add_feature("models_std", REAL);
    result->add_feature("models_min", REAL);
    result->add_feature("models_max", REAL);

    result->add_feature("model_dev_from_mean", REAL);
    result->add_feature("model_dev_from_min", REAL);
    result->add_feature("model_dev_from_max", REAL);

    result->add_feature("model_diff_from_int", REAL);
    result->add_feature("models_range", REAL);
    result->add_feature("models_range_dev_high", REAL);
    result->add_feature("models_range_dev_low", REAL);

    return result;
}

inline float sqr(float f) { return f * f; }

distribution<float>
Gated_Blender::
get_conf_features(int model,
                  const distribution<float> & model_outputs,
                  const distribution<float> & target_singular,
                  const Target_Stats & target_stats) const
{
    distribution<float> result;

    result.push_back(1.0);  // bias

    float real_prediction = model_outputs[model];
    result.push_back(real_prediction);

    result.insert(result.end(),
                  target_singular.begin(), target_singular.end());
    //result.insert(result.end(),
    //              model_outputs.begin(), model_outputs.end());

    for (unsigned i = 0;  i < recomposition_sizes.size();  ++i) {
        int nr = recomposition_sizes[i];

        // TODO: this is unnecessarily done for each of the models
        distribution<float> reconst;
        if (!data->decomposition) reconst = model_outputs;
        else reconst
                 = data->decomposition
                 ->recompose(model_outputs, target_singular, nr);

        //cerr << "reconstitution for order " << nr << " model "
        //     << model << ": in " << model_outputs[model]
        //     << " out: " << reconst[model] << endl;

        result.push_back(reconst[model] - model_outputs[model]);
        result.push_back(abs(reconst[model] - model_outputs[model]));
        result.push_back(sqr(reconst[model] - model_outputs[model]));
        result.push_back((reconst - model_outputs).two_norm());
    }

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
conf(const ML::distribution<float> & models,
     const ML::distribution<float> & target_singular,
     const Target_Stats & target_stats) const
{
    int nm = models.size();

    // For each model, calculate a confidence
    distribution<float> result(nm);

    for (unsigned i = 0;  i < nm;  ++i) {
        // Skip untrained models
        if (model_coefficients[i].empty()) continue;

        // What would we have predicted for this model?

        distribution<float> model_features
            = get_conf_features(i, models, target_singular, target_stats);

        // Perform linear regression (in prediction mode)
        float output = model_features.dotprod(model_coefficients[i]);

        // Link function to change into a probability
        float prob = apply_link_inverse(output, link_function);

        result[i] = prob;
    }

    return result;
}

boost::shared_ptr<Dense_Feature_Space>
Gated_Blender::
blend_feature_space() const
{
    boost::shared_ptr<Dense_Feature_Space> result
        (new Dense_Feature_Space());

    result->add_feature("bias", REAL);

    result->add_feature("min_model", REAL);
    result->add_feature("max_model", REAL);
    result->add_feature("avg_model_chosen", REAL);

    result->add_feature("min_weighted", REAL);
    result->add_feature("max_weighted", REAL);
    result->add_feature("avg_weighted", REAL);
    result->add_feature("weighted_avg", REAL);

    for (unsigned i = 0;  i < data->nm();  ++i) {
        if (data->models[i].rank >= num_models_to_train)
            continue;
        string s = data->model_names[i];
        result->add_feature(s + "_output", REAL);
        result->add_feature(s + "_conf", REAL);
        result->add_feature(s + "_weighted", REAL);
        result->add_feature(s + "_dev_from_mean", REAL);
        result->add_feature(s + "_diff_from_int", REAL);

        for (unsigned i = 0;  i < recomposition_sizes.size();  ++i) {
            string s2 = format("%s_recomp_error_%d", s.c_str(),
                               recomposition_sizes[i]);
            result->add_feature(s2, REAL);
            result->add_feature(s2 + "_abs", REAL);
            result->add_feature(s2 + "_sqr", REAL);
        }
    }

    for (unsigned i = 0;  i < recomposition_sizes.size();  ++i) {
        string s = format("recomp_error_%d", recomposition_sizes[i]);
        result->add_feature(s + "_rmse", REAL);
    }

    //for (unsigned i = 0;  i < data->nm();  ++i)
    //    result->add_feature(data->model_names[i], REAL);

    for (unsigned i = 0;  i < data->singular_targets[0].size();  ++i)
        result->add_feature(format("pc%03d", i), REAL);

    result->add_feature("chosen_conf_min", REAL);
    result->add_feature("chosen_conf_max", REAL);
    result->add_feature("chosen_conf_avg", REAL);
    result->add_feature("highest_chosen_output", REAL);
    result->add_feature("highest_chosen_weighted", REAL);

    result->add_feature("model_avg", REAL);
    result->add_feature("chosen_model_avg", REAL);

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
Gated_Blender::
get_blend_features(const distribution<float> & model_outputs,
                   const distribution<float> & model_conf,
                   const distribution<float> & target_singular,
                   const Target_Stats & target_stats) const
{
    distribution<float> result;

    result.push_back(1.0);  // bias

    result.push_back(model_outputs.min());
    result.push_back(model_outputs.max());

    float avg_model_chosen
        = model_outputs.dotprod(model_conf != 0.0)
        / (model_conf != 0.0).count();

    result.push_back(avg_model_chosen);

    distribution<float> weighted = model_outputs * model_conf;
    result.push_back(weighted.min());
    result.push_back(weighted.max());
    result.push_back(weighted.total() / (model_conf != 0.0).count());
    result.push_back(weighted.total() / model_conf.total());

    distribution<float> dense_model, dense_conf;

    vector<distribution<float> > recompositions;

    for (unsigned i = 0;  i < recomposition_sizes.size();  ++i) {
        int nr = recomposition_sizes[i];
        distribution<float> reconst;
        if (!data->decomposition) reconst = model_outputs;
        else reconst
                 = data->decomposition
                 ->recompose(model_outputs, target_singular, nr);
        recompositions.push_back(reconst);
    }

    for (unsigned i = 0;  i < model_outputs.size();  ++i) {
        if (model_coefficients[i].empty()) continue;
        result.push_back(model_outputs[i]);
        result.push_back(model_conf[i]);
        result.push_back(weighted[i]);
        dense_model.push_back(model_outputs[i]);
        dense_conf.push_back(model_conf[i]);

        float real_prediction = model_outputs[i];

        result.push_back((real_prediction - target_stats.mean)
                         / target_stats.std);
        result.push_back
            (std::min(fabs(real_prediction - ceil(real_prediction)),
                      fabs(real_prediction - floor(real_prediction))));
        
        for (unsigned r = 0;  r < recomposition_sizes.size();  ++r) {
            const distribution<float> & reconst = recompositions[r];
            result.push_back(reconst[i] - model_outputs[i]);
            result.push_back(abs(reconst[i] - model_outputs[i]));
            result.push_back(sqr(reconst[i] - model_outputs[i]));
        }
    }

    for (unsigned i = 0;  i < recomposition_sizes.size();  ++i) {
        const distribution<float> & reconst = recompositions[i];
        result.push_back((reconst - model_outputs).two_norm());
    }
    
    //result.insert(result.end(), model_outputs.begin(), model_outputs.end());

    result.insert(result.end(), target_singular.begin(), target_singular.end());
    result.push_back(dense_conf.min());
    result.push_back(dense_conf.max());
    result.push_back(dense_conf.total() / dense_conf.size());

    distribution<bool> ismax = dense_conf == dense_conf.max();

    result.push_back((dense_model * ismax).total() / (ismax.count()));
    result.push_back((dense_model * dense_conf * ismax).total()
                     / (ismax.count()));

    result.push_back(model_outputs.total() / model_outputs.size());
    result.push_back(dense_model.total() / dense_model.size());

    result.push_back(target_stats.mean);
    result.push_back(target_stats.std);
    result.push_back(target_stats.min);
    result.push_back(target_stats.max);

    result.push_back(target_stats.max - target_stats.min);
    result.push_back((target_stats.max - target_stats.mean) / target_stats.std);
    result.push_back((target_stats.mean - target_stats.min) / target_stats.std);

    result.push_back(target_stats.mean - avg_model_chosen);
    result.push_back(abs(target_stats.mean - avg_model_chosen));


    return result;
}

float
Gated_Blender::
predict(const ML::distribution<float> & models) const
{
    distribution<float> target_singular
        = data->apply_decomposition(models);

    bool debug = debug_predict;

    auto_ptr<Guard> guard;

    if (debug) {
        static Lock lock;
        guard.reset(new Guard(lock));
    }

    Target_Stats target_stats(models.begin(), models.end());
    
    int nm = models.size();

    distribution<float> conf = this->conf(models, target_singular,
                                          target_stats);
    
    //float result = models.dotprod(conf) / conf.total();

    //float result = conf.total() * 0.1 * 4.0 + 1.0;
   
    for (unsigned i = 0;  i < nm && debug;  ++i) {
        if (conf[i] == 0.0) continue;
        cerr << "model " << i << ": pred " << models[i] << " conf "
             << conf[i] << endl;
    }

    distribution<float> blend_features
        = get_blend_features(models, conf, target_singular, target_stats);

#if 0
    Link_Function blend_link_function
        = (target == AUC ? LOGIT : LINEAR);

    float result
        = apply_link_inverse(blend_features.dotprod(blend_coefficients),
                             blend_link_function);
#endif

    blend_features.push_back(correct_prediction);

    boost::shared_ptr<Mutable_Feature_Set> features
        = blender_fs->encode(blend_features);

    if (dump_predict_features != "") {
        Guard guard(predict_feature_lock);
        predict_feature_file << blender_fs->print(*features) << endl;
    }

    //ML::Output_Encoding encoding
    //    = blender->output_encoding();
    //cerr << "output encoding is " << encoding << endl;

    float result;
    if (target == AUC)
        result = blender->predict(1, *features);
    else result = blender->predict(0, *features);

    if (debug) cerr << "result before scaling = "
                    << result << endl;

    if (target == RMSE) {
        //result = result * 2.5 + 3.0;
        if (result < -1.0) result = -1.0;
        if (result >  1.0) result = 1.0;
    }

    if (debug) cerr << "result = " << result << " correct = "
                    << correct_prediction << endl;

    return result;
}

std::string
Gated_Blender::
explain(const ML::distribution<float> & models) const
{
    Target_Stats target_stats(models.begin(), models.end());

    distribution<float> target_singular
        = data->apply_decomposition(models);
    
    distribution<float> conf = this->conf(models, target_singular,
                                          target_stats);
    
    distribution<float> blend_features
        = get_blend_features(models, conf, target_singular, target_stats);

    blend_features.push_back(correct_prediction);

    boost::shared_ptr<Mutable_Feature_Set> features
        = blender_fs->encode(blend_features);

    ML::Explanation explanation
        = blender->explain(*features, target == AUC);

    return explanation.print();
}
