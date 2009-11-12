/* deep_net_blender.cc
   Jeremy Barnes, 30 October 2009
   Copyright (c) 2009 Jeremy Barnes.  All rights reserved.

   Blender based upon training of a deep neural network.
*/

#include "deep_net_blender.h"
#include <boost/progress.hpp>
#include <boost/bind.hpp>
#include "boosting/worker_task.h"
#include "utils/info.h"
#include "utils/guard.h"
#include "arch/timers.h"
#include "dnae_decomposition.h"


using namespace std;
using namespace ML;


namespace ML {

/*****************************************************************************/
/* AUGMENTED_DEEP_NET_UPDATES                                                */
/*****************************************************************************/

Augmented_Deep_Net_Updates::
Augmented_Deep_Net_Updates()
{
}

Augmented_Deep_Net_Updates::
Augmented_Deep_Net_Updates(const Augmented_Deep_Net & net)
    : dnae(net.dnae),
      supervised(net.supervised)
{
}

Augmented_Deep_Net_Updates &
Augmented_Deep_Net_Updates::
operator += (const Augmented_Deep_Net_Updates & updates)
{
    dnae.values += updates.dnae.values;
    supervised.values += updates.supervised.values;
    return *this;
}


/*****************************************************************************/
/* AUGMENTED_DEEP_NET                                                        */
/*****************************************************************************/

Augmented_Deep_Net::
Augmented_Deep_Net()
    : dnae("dnae"), supervised("supervised")
{
}

void
Augmented_Deep_Net::
init(const Auto_Encoder_Stack & dnae, int nfeatures,
     int nhidden, int noutputs, Transfer_Function_Type transfer,
     Thread_Context & context)
{
    this->dnae = dnae;

    // Create the combined model: hidden layer
    supervised.add(new Dense_Layer<float>("hidden", nfeatures + dnae.outputs(),
                                          nhidden, TF_TANH, MV_NONE, context));

    // Output layer
    supervised.add(new Dense_Layer<float>("output", nhidden /* inputs */,
                                          1 /* outputs */, TF_TANH, MV_NONE,
                                          context));
}

float
Augmented_Deep_Net::
predict(const ML::distribution<float> & models,
        const distribution<float> & features) const
{
    distribution<float> dnae_output
        = dnae.apply(0.8 * models);

    dnae_output.insert(dnae_output.end(), features.begin(), features.end());

    return 1.25 * supervised.apply(dnae_output)[0];
}

void
Augmented_Deep_Net::
update(const Augmented_Deep_Net_Updates & updates,
       double learning_rate)
{
    
    dnae.parameters().update(updates.dnae, learning_rate);
    supervised.parameters().update(updates.supervised, learning_rate);
}

std::pair<double, double>
Augmented_Deep_Net::
train_example(const distribution<float> & model_input,
              const distribution<float> & extra_features,
              float label,
              Augmented_Deep_Net_Updates & updates) const
{
    /* fprop - dnae */
    
    size_t dnae_temp_space_required
        = dnae.fprop_temporary_space_required();

    float dnae_temp_space[dnae_temp_space_required];

    distribution<float> dnae_inputs = model_input;
    distribution<float> dnae_outputs
        = dnae.fprop(dnae_inputs, dnae_temp_space, dnae_temp_space_required);

    /* fprop - supervised */
    distribution<float> sup_inputs = dnae_outputs;
    sup_inputs.extend(extra_features);
    
    size_t sup_temp_space_required
        = supervised.fprop_temporary_space_required();

    float sup_temp_space[sup_temp_space_required];

    distribution<float> sup_outputs
        = supervised.fprop(sup_inputs, sup_temp_space, sup_temp_space_required);

    /* errors */
    distribution<float> errors = ((0.8 * label) - sup_outputs);
    double error = errors.dotprod(errors);
    distribution<float> sup_derrors = -2.0 * errors;

    /* bprop - supervised */
    distribution<float> dnae_derrors
        = supervised.bprop(sup_inputs,
                           sup_outputs,
                           sup_temp_space, sup_temp_space_required,
                           sup_derrors,
                           updates.supervised,
                           1.0);

    /* Take the errors for the part of the dnae stack */
    dnae_derrors.resize(dnae.outputs());

    /* bprop - dnae */
    dnae.bprop(dnae_inputs, dnae_outputs,
               dnae_temp_space, dnae_temp_space_required,
               dnae_derrors,
               updates.dnae,
               1.0);

    return make_pair(sqrt(error), sup_outputs[0]);
}

struct Train_Deep_Net_Examples_Job {

    const Augmented_Deep_Net & net;
    const std::vector<distribution<float> > & model_outputs;
    const std::vector<distribution<float> > & features;
    const std::vector<float> & labels;
    Thread_Context & thread_context;
    const vector<int> & examples;
    int first;
    int last;
    Augmented_Deep_Net_Updates & updates;
    vector<float> & outputs;
    double & total_rmse;
    Lock & updates_lock;
    boost::progress_display * progress;
    int verbosity;

    Train_Deep_Net_Examples_Job(const Augmented_Deep_Net & net,
                                const std::vector<distribution<float> > & model_outputs,
                                const std::vector<distribution<float> > & features,
                                const std::vector<float> & labels,
                                Thread_Context & thread_context,
                                const vector<int> & examples,
                                int first, int last,
                                Augmented_Deep_Net_Updates & updates,
                                vector<float> & outputs,
                                double & total_rmse,
                                Lock & updates_lock,
                                boost::progress_display * progress,
                                int verbosity)
        : net(net), model_outputs(model_outputs), features(features),
          labels(labels),
          thread_context(thread_context), examples(examples),
          first(first), last(last),
          updates(updates),
          outputs(outputs),
          total_rmse(total_rmse), updates_lock(updates_lock),
          progress(progress), verbosity(verbosity)
    {
    }

    void operator () ()
    {
        Augmented_Deep_Net_Updates local_updates(net);

        double total_rmse_local = 0.0;

        for (unsigned ix = first; ix < last;  ++ix) {
            int x = examples[ix];

            double rmse_contribution;
            double output;

            boost::tie(rmse_contribution, output)
                = net.train_example(model_outputs[x],
                                    features[x],
                                    labels[x],
                                    local_updates);

            outputs[ix] = output;
            total_rmse_local += rmse_contribution;
        }

        Guard guard(updates_lock);
        total_rmse += total_rmse_local;
        updates += local_updates;
        if (progress) progress += (last - first);
    }
};

std::pair<double, double>
Augmented_Deep_Net::
train_iter(const std::vector<distribution<float> > & model_outputs,
           const std::vector<distribution<float> > & features,
           const std::vector<float> & labels,
           ML::Thread_Context & thread_context,
           int minibatch_size, float learning_rate,
           int verbosity,
           float sample_proportion,
           bool randomize_order)
{
    Worker_Task & worker = thread_context.worker();

    int nx = model_outputs.size();

    int microbatch_size = minibatch_size / (num_threads() * 4);
            
    Lock update_lock;

    vector<int> examples;
    for (unsigned x = 0;  x < nx;  ++x) {
        // Randomly exclude some samples
        if (thread_context.random01() >= sample_proportion)
            continue;
        examples.push_back(x);
    }
    
    if (randomize_order) {
        Thread_Context::RNG_Type rng = thread_context.rng();
        std::random_shuffle(examples.begin(), examples.end(), rng);
    }
    
    int nx2 = examples.size();

    double total_mse = 0.0;
    Model_Output outputs;
    outputs.resize(nx2);    

    std::auto_ptr<boost::progress_display> progress;
    if (verbosity >= 3) progress.reset(new boost::progress_display(nx2, cerr));

    for (unsigned x = 0;  x < nx2;  x += minibatch_size) {
                
        Augmented_Deep_Net_Updates updates(*this);
                
        // Now, submit it as jobs to the worker task to be done
        // multithreaded
        int group;
        {
            int parent = -1;  // no parent group
            group = worker.get_group(NO_JOB, "dump user results task",
                                     parent);
                    
            // Make sure the group gets unlocked once we've populated
            // everything
            Call_Guard guard(boost::bind(&Worker_Task::unlock_group,
                                         boost::ref(worker),
                                         group));
                    
                    
            for (unsigned x2 = x;  x2 < nx2 && x2 < x + minibatch_size;
                 x2 += microbatch_size) {
                        
                Train_Deep_Net_Examples_Job
                    job(*this,
                        model_outputs,
                        features,
                        labels,
                        thread_context,
                        examples,
                        x2,
                        min<int>(nx2,
                                 min(x + minibatch_size,
                                     x2 + microbatch_size)),
                        updates,
                        outputs,
                        total_mse,
                        update_lock,
                        progress.get(),
                        verbosity);

                // Send it to a thread to be processed
                worker.add(job, "backprop job", group);
            }
        }
        
        worker.run_until_finished(group);
        
        update(updates, learning_rate);
    }

    // TODO: calculate AUC score
    distribution<float> test_labels;
    for (unsigned i = 0;  i < nx2;  ++i)
        test_labels.push_back(labels[examples[i]]);

    double auc = outputs.calc_auc(test_labels);

    return make_pair(sqrt(total_mse / nx2), auc);
}

struct Test_Deep_Net_Job {

    const Augmented_Deep_Net & net;
    const vector<distribution<float> > & model_outputs;
    const vector<distribution<float> > & features;
    const vector<float> & labels;
    int first;
    int last;
    const Thread_Context & context;
    Lock & update_lock;
    double & error_rmse;
    vector<float> & outputs;
    boost::progress_display * progress;
    int verbosity;

    Test_Deep_Net_Job(const Augmented_Deep_Net & net,
                      const vector<distribution<float> > & model_outputs,
                      const vector<distribution<float> > & features,
                      const vector<float> & labels,
                      int first, int last,
                      const Thread_Context & context,
                      Lock & update_lock,
                      double & error_rmse,
                      vector<float> & outputs,
                      boost::progress_display * progress,
                      int verbosity)
        : net(net), model_outputs(model_outputs), features(features),
          labels(labels),
          first(first), last(last),
          context(context),
          update_lock(update_lock),
          error_rmse(error_rmse), outputs(outputs),
          progress(progress), verbosity(verbosity)
    {
    }

    void operator () ()
    {
        double local_error_rmse = 0.0;

        for (unsigned x = first;  x < last;  ++x) {
            float output = net.predict(model_outputs[x], features[x]);

            outputs[x] = output;
            local_error_rmse += pow(labels[x] - output, 2);
        }
        
        Guard guard(update_lock);
        error_rmse += local_error_rmse;
        if (progress && verbosity >= 3) (*progress) += (last - first);
    }
};

std::pair<double, double>
Augmented_Deep_Net::
test(const std::vector<distribution<float> > & model_outputs,
     const std::vector<distribution<float> > & features,
     const std::vector<float> & labels,
     ML::Thread_Context & thread_context,
     int verbosity)
{
    Lock update_lock;
    double mse_total = 0.0;

    int nx = model_outputs.size();

    std::auto_ptr<boost::progress_display> progress;
    if (verbosity >= 3) progress.reset(new boost::progress_display(nx, cerr));

    Worker_Task & worker = thread_context.worker();

    Model_Output outputs;
    outputs.resize(nx);
            
    // Now, submit it as jobs to the worker task to be done
    // multithreaded
    int group;
    {
        int parent = -1;  // no parent group
        group = worker.get_group(NO_JOB, "dump user results task",
                                 parent);
        
        // Make sure the group gets unlocked once we've populated
        // everything
        Call_Guard guard(boost::bind(&Worker_Task::unlock_group,
                                     boost::ref(worker),
                                     group));
        
        // 20 jobs per CPU
        int batch_size = nx / (num_cpus() * 20);
        
        for (unsigned x = 0; x < nx;  x += batch_size) {
            
            Test_Deep_Net_Job job(*this, model_outputs, features, labels,
                                  x, min<int>(x + batch_size, nx),
                                  thread_context,
                                  update_lock,
                                  mse_total, outputs,
                                  progress.get(),
                                  verbosity);
            
            // Send it to a thread to be processed
            worker.add(job, "test discrim job", group);
        }
    }
    
    worker.run_until_finished(group);
    
    return make_pair(sqrt(mse_total / nx),
                     outputs.calc_auc
                     (distribution<float>(labels.begin(), labels.end())));
}

#if 0
double
Augmented_Deep_Net::
calc_learning_rate(const std::vector<distribution<float> > & model_outputs,
                   const std::vector<distribution<float> > & features,
                   const std::vector<float> & labels) const
{
    // See http://videolectures.net/eml07_lecun_wia/ and the slides at
    // http://carbon.videolectures.net/2007/pascal/eml07_whistler/lecun_yann/eml07_lecun_wia_01.pdf especially slide 48.

    Augmented_Deep_Net_Updates eigenvector(*this);
    eigenvector.random_fill();

    int nx = model_outputs.size();

    double alpha = 1e-5;

    for (unsigned x = 0;  x < nx;  ++x) {

        Augmented_Deep_Net_Updates example_updates(*this);
        
        train_example(model_outputs[x], features[x], labels[x],
                      example_updates);

        // Now present a perturbed example
        Augmented_Deep_Net perturbed(*this);

        perturbed.update(eigenvector, alpha / eigenvector.two_norm());

        Augmented_Deep_Net_Updates perturbed_updates(*this);

        perturbed.train_example(model_outputs[x], features[x], labels[x],
                                perturbed_updates);

        // Find the difference in the gradient

        Augmented_Deep_Net_Updates difference
            = (perturbed_updates - example_updates);

        // ...
    }
}
#endif

}  // namespace ML


/*****************************************************************************/
/* DEEP_NET_BLENDER                                                          */
/*****************************************************************************/

Deep_Net_Blender::
Deep_Net_Blender()
{
}

Deep_Net_Blender::
~Deep_Net_Blender()
{
}

void
Deep_Net_Blender::
configure(const ML::Configuration & config,
          const std::string & name,
          int random_seed,
          Target target)
{
    this->config = config;
    this->random_seed = random_seed;

    config.require(model_base, "model_base");
}

distribution<float>
Deep_Net_Blender::
get_extra_features(const distribution<float> & model_outputs,
                   const distribution<float> & target_singular,
                   const Target_Stats & stats) const
{
    distribution<float> result;

    result.push_back(model_outputs.min());
    result.push_back(model_outputs.max());

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

    distribution<float> dense_model;

    for (unsigned i = 0;  i < model_outputs.size();  ++i) {
        if (data->models[i].rank >= 10) continue;
        result.push_back(model_outputs[i]);
        dense_model.push_back(model_outputs[i]);

        float real_prediction = model_outputs[i];

        result.push_back((real_prediction - stats.mean)
                         / stats.std);
        result.push_back
            (std::min(fabs(real_prediction - ceil(real_prediction)),
                      fabs(real_prediction - floor(real_prediction))));
        
        for (unsigned r = 0;  r < recomposition_sizes.size();  ++r) {
            const distribution<float> & reconst = recompositions[r];
            result.push_back(reconst[i] - model_outputs[i]);
            result.push_back(abs(reconst[i] - model_outputs[i]));
            result.push_back(pow(reconst[i] - model_outputs[i], 2));
        }
    }

    for (unsigned i = 0;  i < recomposition_sizes.size();  ++i) {
        const distribution<float> & reconst = recompositions[i];
        result.push_back((reconst - model_outputs).two_norm());
    }
    
    //result.insert(result.end(), model_outputs.begin(), model_outputs.end());

    //result.insert(result.end(), target_singular.begin(), target_singular.end());

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



void
Deep_Net_Blender::
init(const Data & data,
     const ML::distribution<float> & example_weights)
{
    this->data = &data;
    
    // Reconstitute the base model
    boost::shared_ptr<Decomposition> loaded;
    const DNAE_Decomposition & decomp
        = dynamic_cast <const DNAE_Decomposition & >
        (*(loaded = Decomposition::load(model_base)));

    int nfeatures = get_extra_features(data.examples[0],
                                       data.singular_targets[0],
                                       data.target_stats[0]).size();

    int nhidden = 50;

    Thread_Context context;
    context.seed(random_seed);

    net.init(decomp.stack, nfeatures, nhidden, 1 /* noutputs */, TF_TANH,
             context);

    float hold_out = 0.2;
    config.get(hold_out, "hold_out");

    Data training_data = data;
    Data testing_data;
    training_data.hold_out(testing_data, hold_out, random_seed);

    // Start training
    vector<distribution<float> > training_samples = training_data.examples;
    vector<distribution<float> > testing_samples = testing_data.examples;

    int nx = training_samples.size();

    for (unsigned i = 0;  i < nx;  ++i)
        training_samples[i] *= 0.8;

    int nxt = testing_samples.size();

    for (unsigned i = 0;  i < nxt;  ++i)
        testing_samples[i] *= 0.8;

    vector<distribution<float> > training_features(nx), testing_features(nxt);

    for (unsigned i = 0;  i < nx;  ++i)
        training_features[i]
            = get_extra_features(training_data.examples[i],
                                 training_data.singular_targets[i],
                                 training_data.target_stats[i]);

    for (unsigned i = 0;  i < nxt;  ++i)
        testing_features[i]
            = get_extra_features(testing_data.examples[i],
                                 testing_data.singular_targets[i],
                                 testing_data.target_stats[i]);
    
    double learning_rate = 0.75;
    int minibatch_size = 512;
    int niter = 50;
    int verbosity = 2;

    Transfer_Function_Type transfer_function = TF_TANH;

    bool randomize_order = true;
    float sample_proportion = 0.8;
    int test_every = 1;

    config.get(learning_rate, "learning_rate");
    config.get(minibatch_size, "minibatch_size");
    config.get(niter, "niter");
    config.get(verbosity, "verbosity");
    config.get(transfer_function, "transfer_function");
    config.get(randomize_order, "randomize_order");
    config.get(sample_proportion, "sample_proportion");
    config.get(test_every, "test_every");

    if (nx == 0)
        throw Exception("can't train on no data");

    // Learning rate is per-example
    learning_rate /= nx;

    // Compensate for the example proportion
    learning_rate /= sample_proportion;

    if (verbosity == 2)
        cerr << "iter  ---- train ----  ---- test -----\n"
             << "         rmse     auc     rmse     auc\n";
    
    for (unsigned iter = 0;  iter < niter;  ++iter) {
        if (verbosity >= 3)
            cerr << "iter " << iter << " training on " << nx << " examples"
                 << endl;
        else if (verbosity >= 2)
            cerr << format("%4d", iter) << flush;
        Timer timer;

        double train_error_rmse, train_error_auc;
        boost::tie(train_error_rmse, train_error_auc)
            = net.train_iter(training_samples, training_features,
                             training_data.targets,
                             context,
                             minibatch_size, learning_rate,
                             verbosity, sample_proportion,
                             randomize_order);
        
        if (verbosity >= 3) {
            cerr << "error of iteration: rmse " << train_error_rmse
                 << " noisy " << train_error_auc << endl;
            if (verbosity >= 3) cerr << timer.elapsed() << endl;
        }
        else if (verbosity == 2)
            cerr << format("  %7.5f %7.5f",
                           train_error_rmse, train_error_auc)
                 << flush;
        
        if (iter % test_every == (test_every - 1)
            || iter == niter - 1) {
            timer.restart();
            double test_error_rmse = 0.0, test_error_auc = 0.0;
                
            if (verbosity >= 3)
                cerr << "testing on " << nxt << " examples"
                     << endl;

            boost::tie(test_error_rmse, test_error_auc)
                = net.test(testing_samples, testing_features,
                           testing_data.targets,
                           context, verbosity);
            
            if (verbosity >= 3) {
                cerr << "testing error of iteration: rmse "
                     << test_error_rmse << " auc " << test_error_auc
                     << endl;
                cerr << timer.elapsed() << endl;
            }
            else if (verbosity == 2)
                cerr << format("  %7.5f %7.5f",
                               test_error_rmse, test_error_auc);
        }
        
        if (verbosity == 2) cerr << endl;
    }
}

float
Deep_Net_Blender::
predict(const ML::distribution<float> & models) const
{
    distribution<float> target_singular
        = data->apply_decomposition(models);

    Target_Stats target_stats(models.begin(), models.end());
    
    distribution<float> features
        = get_extra_features(models, target_singular, target_stats);

    return net.predict(models, features);
}

std::string
Deep_Net_Blender::
explain(const ML::distribution<float> & models) const
{
    return "no explanation";
}

