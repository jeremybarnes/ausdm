/* ausdm.cc                                                       -*- C++ -*-
   Jeremy Barnes, 6 August 2009
   Copyright (c) 2009 Jeremy Barnes.  All rights reserved.

   AusDM entry.
*/

#include "data.h"
#include "blender.h"
#include "decomposition.h"

#include <fstream>
#include <iterator>
#include <iostream>

#include "arch/exception.h"
#include "utils/string_functions.h"
#include "utils/pair_utils.h"
#include "utils/vector_utils.h"
#include "utils/filter_streams.h"
#include "utils/configuration.h"
#include "arch/timers.h"
#include "utils/guard.h"
#include "arch/threads.h"
#include "stats/distribution_ops.h"
#include "utils/parse_context.h"

#include "boosting/worker_task.h"
#include "boosting/thread_context.h"

#include <boost/program_options/cmdline.hpp>
#include <boost/program_options/options_description.hpp>
#include <boost/program_options/positional_options.hpp>
#include <boost/program_options/parsers.hpp>
#include <boost/program_options/variables_map.hpp>
#include <boost/progress.hpp>
#include <boost/tuple/tuple.hpp>
#include <boost/bind.hpp>

#include "stats/moments.h"

using namespace std;
using namespace ML;


struct Predict_Job {
    Model_Output & result;
    int first, last;
    const Blender & blender;
    const Data & data;
    boost::progress_display & progress;
    Lock & progress_lock;

    Predict_Job(Model_Output & result,
                int first, int last,
                const Blender & blender,
                const Data & data,
                boost::progress_display & progress,
                Lock & progress_lock)
        : result(result),
          first(first), last(last), blender(blender),
          data(data), progress(progress), progress_lock(progress_lock)
    {
    }

    void operator () ()
    {
        for (int j = first;  j < last;  ++j) {
            const distribution<float> & model_inputs
                = data.examples[j]->models;

            correct_prediction = data.targets[j];

            float val = blender.predict(model_inputs);
            result.at(j) = val;

            Guard guard(progress_lock);
            ++progress;
        }
    }
};


int main(int argc, char ** argv)
{
    // Filename to dump validation output data to
    string validation_output_file;

    // Filename to dump official testing data to
    string official_output_file;

    // Configuration file to use
    string config_file = "config.txt";

    // Name of blender in config file
    string blender_name;

    // Extra configuration options
    vector<string> extra_config_options;

    // What type of target do we predict?
    string target_type;

    // What is the decomposition?  Either a filename or a type.
    string decomposition_name = "";

    // Which size (S, M, L for Small, Medium and Large)
    string size = "S";

    // Which verbosity level?
    int verbosity = 3;

    // Use rankings of models?
    bool use_rankings = true;

    {
        using namespace boost::program_options;

        options_description config_options("Configuration");

        config_options.add_options()
            ("config-file,c", value<string>(&config_file),
             "configuration file to read configuration options from")
            ("blender-name,n", value<string>(&blender_name),
             "name of blender in configuration file")
            ("extra-config-option", value<vector<string> >(&extra_config_options),
             "extra configuration option=value (can go directly on command line)");

        options_description control_options("Control Options");

        control_options.add_options()
            ("target-type,t", value<string>(&target_type),
             "select target type: auc or rmse")
            ("size,S", value<string>(&size),
             "size: S (small), M (medium) or L (large)")
            ("decomposition", value<string>(&decomposition_name),
             "filename or name of decomposition; empty = none")
            ("validation-output-file,o", value<string>(&validation_output_file),
             "dump validation (blending) output file to the given filename")
            ("official-output-file,O", value<string>(&official_output_file),
             "dump official output file to the given filename")
            ("verbosity,v", value<int>(&verbosity),
             "set verbosity to value")
            ("use-rankings,r", value<bool>(&use_rankings),
             "use ranking features as extra models");

        positional_options_description p;
        p.add("extra-config-option", -1);

        options_description all_opt;
        all_opt
            .add(config_options)
            .add(control_options);

        all_opt.add_options()
            ("help,h", "print this message");
        
        variables_map vm;
        store(command_line_parser(argc, argv)
              .options(all_opt)
              .positional(p)
              .run(),
              vm);
        notify(vm);

        if (vm.count("help")) {
            cout << all_opt << endl;
            return 1;
        }
    }

    Target target;
    if (target_type == "auc") target = AUC;
    else if (target_type == "rmse") target = RMSE;
    else throw Exception("target type " + target_type + " not known");

    if (blender_name == "")
        blender_name = target_type;

    // Load up configuration
    Configuration config;
    if (config_file != "") config.load(config_file);

    vector<string> data_files;
    vector<string> config_overrides;
    
    for (unsigned i = 0;  i < extra_config_options.size();  ++i) {
        if (extra_config_options[i].find('=') != string::npos)
            config_overrides.push_back(extra_config_options[i]);
        else data_files.push_back(extra_config_options[i]);
    }

    // Allow configuration to be overridden on the command line
    config.parse_command_line(config_overrides);
    
    // Load up the data
    Timer timer;

    cerr << "loading data...";

    vector<int> row_ids;
    vector<Model_Output> models_train, models_test;
    vector<string> model_names;
    distribution<float> labels;
    int num_rows_test = 0;
    
    for (unsigned i = 0;  i < data_files.size();  ++i) {
        Parse_Context context("loadbuild/" + data_files[i] + "/"
                              + size + "_" + target_type + "_merge.txt");

        int row = 0;

        Model_Output current_model;

        for (;  context;  ++row) {
            if (i != 0 && row >= row_ids.size())
                context.exception("mismatch in number of rows");

            // Format: row ID target prediction
            context.skip_whitespace();
            int row_id = context.expect_int();
            
            if (i == 0) row_ids.push_back(row_id);
            else if (row_id != row_ids[row])
                context.exception("row ID doesn't match other models");

            context.expect_whitespace();
            float label = context.expect_int();

            if (target == RMSE) label = (label - 3000.0) / 2000.0;
            if (i == 0)
                labels.push_back(label);
            else if (labels[row] != label)
                context.exception("label doesn't match");

            context.expect_whitespace();
            float prediction = context.expect_float();

            if (target == RMSE || true) prediction = (prediction - 3000.0) / 2000.0;
            
            current_model.push_back(prediction);

            context.expect_eol();
        }

        if (i != 0 && row < row_ids.size())
            context.exception("not enough rows");

        model_names.push_back(data_files[i]);
        models_train.push_back(current_model);

        if (use_rankings) {
            // We add two models: the actual output, as well as a sorted version
            // giving the rank.
            
            vector<pair<int, float> > sorted_predictions;
            for (unsigned r = 0;  r < row_ids.size();  ++r)
                sorted_predictions.push_back(make_pair(r, current_model[r]));
            
            sort_on_second_ascending(sorted_predictions);
            
            
            Model_Output current_ranked;
            current_ranked.resize(row_ids.size());
            for (unsigned r = 0;  r < sorted_predictions.size();  ++r) {
                current_ranked[sorted_predictions[r].first]
                    = -1.0 + (r * 2.0 / sorted_predictions.size());
            }
            
            model_names.push_back(data_files[i] + "_rank");
            models_train.push_back(current_ranked);
        }

#if 0
        cerr << "model " << data_files[i] << ": "
             << distribution<float>(current_model.begin(),
                                    current_model.begin() + 10)
             << endl;

        cerr << "model " << data_files[i] << " ranked: "
             << distribution<float>(current_ranked.begin(),
                                    current_ranked.begin() + 10)
             << endl;
#endif
        
        
        Parse_Context context2("loadbuild/" + data_files[i] + "/"
                               + size + "_" + target_type + "_official.txt");

        Model_Output current_model_test;

        row = 0;
        for (;  context2;  ++row) {
            if (i != 0 && row >= num_rows_test)
                context2.exception("mismatch in number of rows");

            // Format: prediction
            context2.skip_whitespace();

            float prediction = context2.expect_float();
            
            if (target == RMSE || true) prediction = (prediction - 3000.0) / 2000.0;
            
            current_model_test.push_back(prediction);
            
            context2.expect_eol();
        }

        if (i != 0 && row < num_rows_test)
            context2.exception("not enough rows");
        num_rows_test = row;

        models_test.push_back(current_model_test);

        if (use_rankings) {
            vector<pair<int, float> > sorted_predictions;
            for (unsigned r = 0;  r < num_rows_test;  ++r)
                sorted_predictions.push_back(make_pair(r, current_model_test[r]));
            
            sort_on_second_ascending(sorted_predictions);
            
            Model_Output current_ranked;
            current_ranked.resize(num_rows_test);
            for (unsigned r = 0;  r < sorted_predictions.size();  ++r) {
                current_ranked[sorted_predictions[r].first]
                    = -1.0 + (r * 2.0 / sorted_predictions.size());
            }

            models_test.push_back(current_ranked);
        }
    }

    if (models_train.size() != models_test.size())
        throw Exception("models are a different size");

    int nm = models_train.size();
    
    Data data_train, data_test;

    data_train.target = data_test.target = target;
    data_train.model_names = data_test.model_names = model_names;
    data_train.example_ids = row_ids;
    for (unsigned i = 0;  i < num_rows_test;  ++i)
        data_test.example_ids.push_back(i);
    data_train.targets = labels;
    data_test.targets.resize(num_rows_test);

    int num_rows = row_ids.size();
    
    for (unsigned i = 0;  i < num_rows;  ++i) {
        distribution<float> ex_models(nm);

        for (unsigned m = 0;  m < nm;  ++m)
            ex_models[m] = models_train[m][i];

        boost::shared_ptr<Data::Example> example
            (new Data::Example(ex_models, labels[i], target));

        data_train.examples.push_back(example);
    }

    data_train.models.resize(nm);

    data_train.calc_scores();

    data_test.models = data_train.models;
    data_test.model_ranking = data_train.model_ranking;

    for (unsigned i = 0;  i < num_rows_test;  ++i) {
        distribution<float> ex_models(nm);
        
        for (unsigned m = 0;  m < nm;  ++m)
            ex_models[m] = models_test[m][i];
        
        boost::shared_ptr<Data::Example> example
            (new Data::Example(ex_models, labels[i], target));

        data_test.examples.push_back(example);
    }

    cerr << "data_train.nm() = " << data_train.nm() << endl;

    boost::shared_ptr<Decomposition> decomposition;
    if (Decomposition::known_type(decomposition_name)) {
        decomposition = Decomposition::create(decomposition_name);
        
        cerr << "training decomposition" << endl;
        decomposition->train(data_train,
                             data_train,
                             config);
        cerr << "done" << endl;
    }
    else if (decomposition_name != "") {
        decomposition = Decomposition::load(decomposition_name);
        decomposition->init(config);
    }

    Model_Output result;

    int nx = data_train.nx();

    if (decomposition) {
        cerr << "applying decomposition" << endl;
        data_train.apply_decomposition(*decomposition);
        cerr << "done." << endl;
        data_test.apply_decomposition(*decomposition);
    }


    int nfolds = 10;

    Thread_Context context;

    // Assign each example to a fold
    distribution<int> example_folds(nx);
    for (unsigned i = 0;  i < nx;  ++i)
        example_folds[i] = context.random01() * nfolds;

    cerr << "data_test.nx = " << data_test.nx() << endl;

    Model_Output train_output;
    train_output.resize(nx);
    Model_Output test_output;
    test_output.resize(data_test.nx());

    for (unsigned fold = 0;  fold < nfolds;  ++fold) {
        cerr << "fold " << fold << " of " << nfolds << endl;

        Data data_train_fold = data_train;
        Data data_test_fold;

        //cerr << "data_train_fold.labels = " << data_train_fold.targets
        //     << endl;

        // Hold out those that are in our fold
        distribution<bool> held_out = (example_folds == fold);
        data_train_fold.hold_out(data_test_fold, held_out);

        cerr << "examples: train " << data_train_fold.nx() << " test "
             << data_test_fold.nx() << endl;

        distribution<float> example_weights(data_train_fold.nx(),
                                            1.0 / data_train_fold.nx());

        boost::shared_ptr<Blender> blender
            = get_blender(config, blender_name, data_train_fold,
                          example_weights, fold, target);
        

        int nxt = data_test_fold.nx();
        
        // Now run the model
        result.resize(nxt);
        
        static Worker_Task & worker = Worker_Task::instance(num_threads() - 1);
        
        cerr << "processing " << nxt << " predictions..." << endl;
        
        boost::progress_display progress(nxt, cerr);
        Lock progress_lock;
        
        // Now, submit it as jobs to the worker task to be done multithreaded
        int group;
        int job_num = 0;
        {
            int parent = -1;  // no parent group
            group = worker.get_group(NO_JOB, "dump user results task", parent);
            
            // Make sure the group gets unlocked once we've populated
            // everything
            Call_Guard guard(boost::bind(&Worker_Task::unlock_group,
                                         boost::ref(worker),
                                         group));
            
            for (unsigned i = 0;  i < nxt;  i += 100, ++job_num) {
                int last = std::min<int>(nxt, i + 100);
                
                // Create the job
                Predict_Job job(result,
                                i, last,
                                *blender,
                                data_test_fold, progress, progress_lock);
                
                // Send it to a thread to be processed
                worker.add(job, "blend job", group);
            }
        }
        
        // Add this thread to the thread pool until we're ready
        worker.run_until_finished(group);
        
        cerr << " done." << endl;
        
        // Put back the training output entries
        int n = 0;
        for (unsigned i = 0;  i < nx;  ++i) {
            if (held_out[i]) {
                if (train_output[i] != 0.0)
                    throw Exception("result written twice");
                train_output[i] = result[n];
                ++n;
            }
        }

        cerr << timer.elapsed() << endl;

        if (official_output_file == "") continue;

        job_num = 0;
        nxt = data_test.nx();

        result.resize(nxt);

        cerr << "calculation " << nxt << " official test outputs" << endl;
        progress.restart(nxt);

        {
            int parent = -1;  // no parent group
            group = worker.get_group(NO_JOB, "dump user results task", parent);
            
            // Make sure the group gets unlocked once we've populated
            // everything
            Call_Guard guard(boost::bind(&Worker_Task::unlock_group,
                                         boost::ref(worker),
                                         group));
            
            for (unsigned i = 0;  i < nxt;  i += 100, ++job_num) {
                int last = std::min<int>(nxt, i + 100);
                
                // Create the job
                Predict_Job job(result,
                                i, last,
                                *blender,
                                data_test, progress, progress_lock);
                
                // Send it to a thread to be processed
                worker.add(job, "blend job", group);
            }
        }
        
        // Add this thread to the thread pool until we're ready
        worker.run_until_finished(group);

        // Accumulate the official test output from all of the folds
        test_output += result / nfolds;
    }

    for (unsigned i = 0;  i < nm;  ++i) {
        float score = models_train[i].calc_score(labels, target);
        cerr << format("%-30s %6.4f", model_names[i].c_str(), score)
             << endl;
    }
    
    float score = train_output.calc_score(labels, target);
    cerr << format("%-30s %6.4f", "combined", score) << endl;

    if (validation_output_file != "") {
        filter_ostream out(validation_output_file);
        for (unsigned i = 0;  i < train_output.size();  ++i)
            out << format("%6d %6d %.1f",
                          row_ids[i],
                          (target == AUC ? (int)labels[i]
                           : (int)(labels[i] * 2000 + 3000)),
                          (target == AUC ? train_output[i] * 1000.0
                           : train_output[i] * 2000.0 + 3000.0)) << endl;
    }
    
    if (official_output_file != "") {
        filter_ostream out(official_output_file);

        cerr << "test_output.size() = " << test_output.size() << endl;

        for (unsigned i = 0;  i < test_output.size();  ++i)
            out << format("%.1f",
                          (target == AUC ? test_output[i] * 1000.0
                           : test_output[i] * 2000.0 + 3000.0)) << endl;
    }
}
