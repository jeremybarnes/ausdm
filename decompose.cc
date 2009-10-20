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

#include "boosting/perceptron_generator.h"


using namespace std;
using namespace ML;

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

    // What type of target do we predict?
    //string target_type;
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
            //("target-type,t", value<string>(&target_type),
            // "select target type: auc or rmse")
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

#if 0
    Target target;
    if (target_type == "auc") target = AUC;
    else if (target_type == "rmse") target = RMSE;
    else throw Exception("target type " + target_type + " not known");

    if (decomposer_name == "")
        decomposer_name = target_type;
#endif

    // Load up configuration
    Configuration config;
    if (config_file != "") config.load(config_file);

    // Allow configuration to be overridden on the command line
    config.parse_command_line(extra_config_options);

    // Load up the data
    Timer timer;

    cerr << "loading data...";

    // The decomposition is entirely unsupervised, and so doesn't use the
    // label at all; thus we can put the training and validation sets together

    // Data[s/m/l][auc/rmse]
    Data data[3][2];

    const char * const size_names[3]   = { "S", "M", "L" };
    const char * const target_names[2] = { "RMSE", "AUC" };
    const char * const set_names[2]    = { "Train", "Score" };

    set<string> model_names;

    for (unsigned i = 0;  i < 2;  ++i) {
        for (unsigned j = 0;  j < 2;  ++j) {
            for (unsigned k = 0;  k < 2;  ++k) {
                string filename = format("download/%s_%s_%s.csv",
                                         size_names[i],
                                         target_names[j],
                                         set_names[k]);
                Data & this_data = data[i][j];

                this_data.load(filename, (Target)j, false /* clear first */);
                
                model_names.insert(this_data.model_names.begin(),
                                   this_data.model_names.end());
            }
        }
    }

    cerr << "done" << endl;

    cerr << model_names.size() << " total models" << endl;

    // Now, we look at all of the column names to get an idea of the input
    // dimensions.  


    // Denoising auto encoder
    // We train a stack of layers, one at a time

    Perceptron::Layer layer;
    
    for (unsigned x = 0;  x < nx;  ++x) {
        // Present this input
        distribution<float> model_input;

        // Add noise
        distribution<float> noisy_model_input
            = add_noise(model_input);

        // Apply the layer
        distribution<float> hidden_rep
            = layer.apply(noisy_model_input);

        // Reconstruct the input
        distribution<float> denoised_input
            = layer.inverse(hidden_rep);

        // Error signal
        distribution<float> error
            = model_input - denoised_input;

        // Propagation of error
    }
    
}
