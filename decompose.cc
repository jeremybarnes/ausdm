/* ausdm.cc                                                       -*- C++ -*-
   Jeremy Barnes, 6 August 2009
   Copyright (c) 2009 Jeremy Barnes.  All rights reserved.

   AusDM entry.
*/

#include "data.h"

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

using namespace std;
using namespace ML;

int main(int argc, char ** argv)
{
    // Filename to dump decomposition to
    string output_file;

    // Configuration file to use
    string config_file = "config.txt";

    // Name of blender in config file
    string decomposer_name;

    // Extra configuration options
    vector<string> extra_config_options;

    // What type of target do we predict?
    string target_type;
    {
        using namespace boost::program_options;

        options_description config_options("Configuration");

        config_options.add_options()
            ("config-file,c", value<string>(&config_file),
             "configuration file to read configuration options from")
            ("decomposer-name,n", value<string>(&decomposer_name),
             "name of decomposer in configuration file")
            ("extra-config-option", value<vector<string> >(&extra_config_options),
             "extra configuration option=value (can go directly on command line)");

        options_description control_options("Control Options");

        control_options.add_options()
            ("target-type,t", value<string>(&target_type),
             "select target type: auc or rmse")
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

    if (decomposer_name == "")
        decomposer_name = target_type;

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


    Model_Output result;

    Data data_train;
    data_train.load("download/S_" + targ_type_uc + "_Train.csv", target);

    Data data_test;
    data_test.load("download/S_" + targ_type_uc + "_Score.csv", target);
    


    if (hold_out_data > 0.0) {
        double mean = Stats::mean(trial_scores.begin(), trial_scores.end());
        double std = Stats::std_dev(trial_scores.begin(), trial_scores.end(),
                                    mean);
        
        cout << "scores: " << trial_scores << endl;
        cout << format("%6.4f +/- %6.4f", mean, std) << endl;
    }

    if (output_file != "" && num_trials == 1) {
        filter_ostream out(output_file);
        for (unsigned i = 0;  i < result.size();  ++i)
            out << format("%.6f", result[i] * 1000.0) << endl;
    }
}
