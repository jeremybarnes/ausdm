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
    // Filename to dump output data to
    string output_file;

    // Configuration file to use
    string config_file = "config.txt";

    // Extra configuration options
    vector<string> extra_config_options;

    // Do we perform a fake test (with held-out data)?
    float hold_out_data = 0.0;

    // Tranche specification
    string tranches = "1";

    {
        using namespace boost::program_options;

        options_description config_options("Configuration");

        config_options.add_options()
            ("config-file,c", value<string>(&config_file),
             "configuration file to read configuration options from")
            ("extra-config-option", value<vector<string> >(&extra_config_options),
             "extra configuration option=value (can go directly on command line)");

        options_description control_options("Control Options");

        control_options.add_options()
            ("hold-out-data,T", value<float>(&hold_out_data),
             "run a local test and score on held out data")
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

    // Load up configuration
    Configuration config;
    if (config_file != "") config.load(config_file);

    // Allow configuration to be overridden on the command line
    config.parse_command_line(extra_config_options);

    // Load up the data
    Timer timer;

    cerr << "loading data...";

    Data data_rmse_train, data_auc_train;
    data_rmse_train.load("download/S_RMSE_Train.csv", RMSE);
    data_auc_train.load("download/S_AUC_Train.csv", AUC);

    Data data_rmse_test, data_auc_test;
    if (hold_out_data > 0.0) {
        data_rmse_train.hold_out(data_rmse_test, hold_out_data);
        data_auc_train.hold_out(data_auc_test, hold_out_data);
    }
    else {
        data_rmse_test.load("download/S_RMSE_Score.csv");
        data_auc_test.load("download/S_AUC_Score.csv");
    }

    // Calculate the scores necessary for the job
    data_rmse_train.calc_scores();
    data_auc_train.calc_scores();

    // Train average of top twenty models
    


    Data::Model_Output result;

    cerr << " done." << endl;
    
    cerr << timer.elapsed() << endl;
}
