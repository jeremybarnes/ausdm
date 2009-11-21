/* multiple_regression_blender.cc
   Jeremy Barnes, 2 November 2009
   Copyright (c) 2009 Jeremy Barnes.  All rights reserved.

   Blender for multiple regression models.
*/

#include "multiple_regression_blender.h"
#include "utils.h"
#include "boosting/worker_task.h"
#include "utils/guard.h"

#include <boost/bind.hpp>
#include <boost/progress.hpp>

using namespace std;
using namespace ML;


/*****************************************************************************/
/* MULTIPLE_REGRESSION_BLENDER                                               */
/*****************************************************************************/

Multiple_Regression_Blender::
Multiple_Regression_Blender()
    : decomposition(0)
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
    
    link_function = (target == AUC ? LOGIT : LINEAR);
    config.find(link_function, "link_function");

    num_iter = 200;
    num_examples = 5000;
    num_features = 100;
    ridge_regression = true;

    use_decomposition_features = true;
    use_extra_features = true;

    config.find(num_iter, "num_iter");
    config.find(num_examples, "num_examples");
    config.find(num_features, "num_features");
    config.find(ridge_regression, "ridge_regression");
    config.find(use_decomposition_features, "use_decomposition_features");
    config.find(use_extra_features, "use_extra_features");

    this->random_seed = random_seed;
    this->target = target;
}

distribution<float>
Multiple_Regression_Blender::
train_model(const Data & data,
            Thread_Context & thread_context) const
{
    if (data.nx() == 0)
        throw Exception("can't train with no data");

    int nf = get_features(data.examples[0]->models,
                          data.examples[0]->decomposed,
                          data.examples[0]->stats).size();

    // Number of examples to take
    int nx_me = std::min(data.nx(), num_examples);

    // Number of models to take
    int nf_me = std::min(nf, num_features);
    
    int nx = data.nx();

    // Choose models randomly
    set<int> features_done;
    while (features_done.size() < nf_me) {
        features_done.insert(thread_context.random01() * nf);
    }

    vector<int> kept_features(features_done.begin(), features_done.end());

    // Choose examples randomly
    set<int> examples_done;
    while (examples_done.size() < nx_me) {
        examples_done.insert(thread_context.random01() * nx);
    }

    vector<int> examples(examples_done.begin(), examples_done.end());

    typedef double Float;
    distribution<Float> correct(nx_me);
    boost::multi_array<Float, 2> outputs(boost::extents[nf_me + 1][nx_me]);
    distribution<Float> w(nx_me, 1.0);

    for (unsigned i = 0;  i < nx_me;  ++i) {
        distribution<float> features
            = get_features(data.examples[examples[i]]->models,
                           data.examples[examples[i]]->decomposed,
                           data.examples[examples[i]]->stats);

        for (unsigned j = 0;  j < nf_me;  ++j)
            outputs[j][i] = features[kept_features[j]];

        outputs[nf_me][i] = 1.0;  // bias

        correct[i] = data.targets[examples[i]];
        if (target == AUC)
            correct[i] = (correct[i] == 1.0);
    }
    
    distribution<Float> trained_params
        = perform_irls(correct, outputs, w, link_function, ridge_regression);

    distribution<float> result(nf + 1);

    for (unsigned i = 0;  i < kept_features.size();  ++i)
        result[kept_features[i]] = trained_params[i];
    result[nf] = trained_params.back();  // bias

    return result;
}

namespace {

struct Train_Model_Job {
    const Multiple_Regression_Blender & blender;
    distribution<double> & result;
    Lock & lock;
    const Data & train;
    const Data & test;
    int random_seed;
    boost::progress_display & progress;

    Train_Model_Job(const Multiple_Regression_Blender & blender,
                    distribution<double> & result,
                    Lock & lock,
                    const Data & train,
                    const Data & test,
                    int random_seed,
                    boost::progress_display & progress)
        : blender(blender), result(result), lock(lock), train(train),
          test(test), random_seed(random_seed), progress(progress)
    {
    }

    void operator () ()
    {
        Thread_Context context;
        context.seed(random_seed);
        distribution<float> params = blender.train_model(train, context);
        Guard guard(lock);
        result += params;
        ++progress;
    }
};

} // file scope

void
Multiple_Regression_Blender::
init(const Data & training_data,
     const ML::distribution<float> & example_weights)
{
    Data train = training_data, test;

    bool squashed = (link_function == LINEAR);

    this->decomposition = training_data.decomposition;
    this->model_stats = training_data.models;
    this->model_names = training_data.model_names;

    if (decomposition)
        recomposition_orders = decomposition->recomposition_orders();

    int nf = get_features(training_data.examples[0]->models,
                          training_data.examples[0]->decomposed,
                          training_data.examples[0]->stats).size();

    coefficients.clear();
    coefficients.resize(squashed ? 1 : num_iter, distribution<double>(nf + 1));
    
    Thread_Context context;
    context.seed(random_seed);

    Lock lock;

    static Worker_Task & worker = Worker_Task::instance(num_threads() - 1);
    
    cerr << "training " << num_iter << " models" << endl;
    boost::progress_display progress(num_iter, cerr);

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

        for (unsigned i = 0;  i < num_iter;  ++i) {
            worker.add
                (Train_Model_Job(*this,
                                 (squashed
                                  ? coefficients[0] : coefficients[i]),
                                 lock, train, test,
                                 context.random(), progress),
                 "train model job",
                 group);
        }
    }

    // Add this thread to the thread pool until we're ready
    worker.run_until_finished(group);

    if (squashed) coefficients[0] /= num_iter;

    distribution<double> totals(nf + 1);
    for (unsigned i = 0;  i < coefficients.size();  ++i)
        totals += coefficients[i];

    for (unsigned i = 0;  i <= nf;  ++i) {
        cerr << format("%-30s %9.5f",
                       (i == nf ? "bias" : model_names[i].c_str()),
                       totals[i])
             << endl;
    }
}

distribution<float>
Multiple_Regression_Blender::
get_features(const ML::distribution<float> & models) const
{
    distribution<float> decomposed;
    if (decomposition)
        decomposed = decomposition->decompose(models);
    Target_Stats stats(models.begin(), models.end());

    return get_features(models, decomposed, stats);
}

distribution<float>
Multiple_Regression_Blender::
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
    
    result.push_back(model_outputs.total() / model_outputs.size());
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
Multiple_Regression_Blender::
predict(const ML::distribution<float> & models) const
{
    distribution<float> features = get_features(models);
    features.push_back(1.0);

    double total = 0.0;

    for (unsigned i = 0;  i < coefficients.size();  ++i) {
        float output = features.dotprod(coefficients[i]);
        // Link function to change into a probability
        float prob = apply_link_inverse(output, link_function);

        total += prob;
    }

    return total / coefficients.size();
}

std::string
Multiple_Regression_Blender::
explain(const ML::distribution<float> & models) const
{
    return "";
}

