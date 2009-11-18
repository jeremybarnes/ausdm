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

    config.find(num_iter, "num_iter");
    config.find(num_examples, "num_examples");
    config.find(num_features, "num_features");
    config.find(ridge_regression, "ridge_regression");

    this->random_seed = random_seed;
    this->target = target;
}

distribution<float>
Multiple_Regression_Blender::
train_model(const Data & data,
            Thread_Context & thread_context) const
{
    // Number of examples to take
    int nx_me = std::min(data.nx(), num_examples);

    // Number of models to take
    int nm_me = std::min(data.nm(), num_features);
    
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
        = perform_irls(correct, outputs, w, link_function, ridge_regression);

    //cerr << "trained_params = " << trained_params << endl;

    distribution<float> result(nm);

    for (unsigned i = 0;  i < models.size();  ++i)
        result[models[i]] = trained_params[i];

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

    coefficients.clear();
    coefficients.resize(train.nm());
    
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
            worker.add(Train_Model_Job(*this, coefficients, lock, train, test,
                                       context.random(), progress),
                       "train model job",
                       group);
        }
    }

    // Add this thread to the thread pool until we're ready
    worker.run_until_finished(group);

    coefficients /= num_iter;
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

