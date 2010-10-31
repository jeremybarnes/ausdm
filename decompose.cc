/* ausdm.cc                                                        -*- C++ -*-
   Jeremy Barnes, 6 August 2009
   Copyright (c) 2009 Jeremy Barnes.  All rights reserved.

   AusDM entry.
*/

#include "data.h"

#include <fstream>
#include <iterator>
#include <iostream>

#include "jml/arch/exception.h"
#include "jml/utils/string_functions.h"
#include "jml/utils/pair_utils.h"
#include "jml/utils/vector_utils.h"
#include "jml/utils/filter_streams.h"
#include "jml/utils/configuration.h"
#include "jml/arch/timers.h"
#include "jml/utils/info.h"
#include "jml/utils/guard.h"
#include "jml/arch/threads.h"

#include <boost/program_options/cmdline.hpp>
#include <boost/program_options/options_description.hpp>
#include <boost/program_options/positional_options.hpp>
#include <boost/program_options/parsers.hpp>
#include <boost/program_options/variables_map.hpp>

#include "decomposition.h"


using namespace std;
using namespace ML;
using namespace ML::DB;


int main(int argc, char ** argv)
{
    // Filename to dump decomposition to
    string output_file;

    // Configuration file to use
    string config_file = "config.txt";

    // Name of decomposer in config file
    string decomposer_name;

    // Extra configuration options
    vector<string> extra_config_options;

    // What type of decomposition?
    std::string decomposition_type;

    // What type of target do we predict?
    string target_type;

    // Which size (S, M, L for Small, Medium and Large)
    string size = "S";

    {
        using namespace boost::program_options;

        options_description config_options("Configuration");

        config_options.add_options()
            ("config-file,c", value<string>(&config_file),
             "configuration file to read configuration options from")
            ("decomposer-name,n", value<string>(&decomposer_name),
             "name of decomposer in configuration file")
            ("decomposer-type,T", value<string>(&decomposition_type),
             "type of decomposition to train")
            ("extra-config-option", value<vector<string> >(&extra_config_options),
             "extra configuration option=value (can go directly on command line)");

        options_description control_options("Control Options");

        control_options.add_options()
            ("target-type,t", value<string>(&target_type),
             "select target type: auc or rmse")
            ("size,S", value<string>(&size),
             "size: S (small), M (medium) or L (large)")
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

    Data data_train;
    data_train.load("download/" + size + "_" + targ_type_uc + "_Train.csv.gz",
                    target);
    
    Data data_test;
    data_test.load("download/" + size + "_" + targ_type_uc + "_Score.csv.gz",
                   target);

    // NOTE: these are reversed on purpose; we don't want to overfit our
    // training data
    const Data & training_data = data_test;
    const Data & testing_data  = data_train;

    boost::shared_ptr<Decomposition> decomposition
        = Decomposition::create(decomposition_type);
    
    decomposition->train(training_data, testing_data, config);

    if (output_file != "")
        decomposition->save(output_file);

    cerr << timer.elapsed() << endl;
}
