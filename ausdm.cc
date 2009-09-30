/* ausdm.cc                                                       -*- C++ -*-
   Jeremy Barnes, 6 August 2009
   Copyright (c) 2009 Jeremy Barnes.  All rights reserved.

   AusDM entry.
*/

#include "data.h"
#include "blender.h"

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

#include "boosting/worker_task.h"

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
        : result(result), first(first), last(last), blender(blender),
          data(data), progress(progress), progress_lock(progress_lock)
    {
    }

    void operator () ()
    {
        for (int j = first;  j < last;  ++j) {
            distribution<float> model_inputs(data.models.size());
            for (unsigned i = 0;  i < data.models.size();  ++i) {
                model_inputs[i] = data.models[i][j];
            }

            float val = blender.predict(model_inputs);
            result.at(j) = val;

            Guard guard(progress_lock);
            ++progress;
        }
    }
};


int main(int argc, char ** argv)
{
    // Filename to dump output data to
    string output_file;

    // Configuration file to use
    string config_file = "config.txt";

    // Name of blender in config file
    string blender_name = "blender";

    // Extra configuration options
    vector<string> extra_config_options;

    // Do we perform a fake test (with held-out data)?
    float hold_out_data = 0.0;

    // What type of target do we predict?
    string target_type;

    // How many cross-validation trials do we perform?
    int num_trials = 1;

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
            ("hold-out-data,T", value<float>(&hold_out_data),
             "run a local test and score on held out data")
            ("target-type,t", value<string>(&target_type),
             "select target type: auc or rmse")
            ("num-trials,r", value<int>(&num_trials),
             "select number of trials to perform")
            ("output-file,o",
             value<string>(&output_file),
             "dump output file to the given filename");

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

    // Load up configuration
    Configuration config;
    if (config_file != "") config.load(config_file);

    // Allow configuration to be overridden on the command line
    config.parse_command_line(extra_config_options);

    // Load up the data
    Timer timer;

    cerr << "loading data...";

    string targ_type_uc;
    if (target == AUC) targ_type_uc = "AUC";
    else if (target == RMSE) targ_type_uc = "RMSE";
    else throw Exception("unknown target type");

    vector<double> trial_scores;

    if (hold_out_data == 0.0 && num_trials > 1)
        throw Exception("need to hold out data for multiple trials");

    for (unsigned trial = 0;  trial < num_trials;  ++trial) {
        if (num_trials > 1) cerr << "trial " << trial << endl;

        int rand_seed = hold_out_data > 0.0 ? 1 + trial : 0;

        Data data_train;
        data_train.load("download/S_" + targ_type_uc + "_Train.csv", target);

        Data data_test;
        if (hold_out_data > 0.0)
            data_train.hold_out(data_test, hold_out_data, rand_seed);
        else data_test.load("download/S_" + targ_type_uc + "_Score.csv", target);
        
        // Calculate the scores necessary for the job
        data_train.calc_scores();
        
        boost::shared_ptr<Blender> blender
            = get_blender(config, blender_name, data_train, rand_seed);
        
        int np = data_test.targets.size();
        
        // Now run the model
        Model_Output result;
        result.resize(np);
        
        static Worker_Task & worker = Worker_Task::instance(num_threads() - 1);
        
        cerr << "processing " << np << " predictions..." << endl;
        
        boost::progress_display progress(np, cerr);
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
            
            for (unsigned i = 0;  i < np;  i += 100, ++job_num) {
                int last = std::min<int>(np, i + 100);
                
                // Create the job
                Predict_Job job(result,
                                i, last,
                                *blender, data_test, progress, progress_lock);
                
                // Send it to a thread to be processed
                worker.add(job, "blend job", group);
            }
        }
        
        // Add this thread to the thread pool until we're ready
        worker.run_until_finished(group);
        
        cerr << " done." << endl;
        
        cerr << timer.elapsed() << endl;


        if (hold_out_data > 0.0) {
            double score = result.calc_score(data_test.targets, target);
            cerr << format("%0.4f", score) << endl;
            trial_scores.push_back(score);
        }
    }

    if (hold_out_data > 0.0) {
        double mean = Stats::mean(trial_scores.begin(), trial_scores.end());
        double std = Stats::std_dev(trial_scores.begin(), trial_scores.end(),
                                    mean);
        
        cout << format("%6.4f +/- %6.4f", mean, std) << endl;
    }
}
