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

#include "dnae_decomposition.h"
#include "jml/neural/auto_encoder_trainer.h"
#include "jml/neural/twoway_layer.h"

using namespace std;
using namespace ML;
using namespace ML::DB;


int main(int argc, char ** argv)
{
    // Configuration file to use
    string config_file = "config.txt";

    // Extra configuration options
    vector<string> extra_config_options;
    {
        using namespace boost::program_options;

        options_description config_options("Configuration");

        config_options.add_options()
            ("config-file,c", value<string>(&config_file),
             "configuration file to read configuration options from")
            ("extra-config-option", value<vector<string> >(&extra_config_options),
             "extra configuration option=value (can go directly on command line)");

        positional_options_description p;
        p.add("extra-config-option", -1);

        options_description all_opt;
        all_opt
            .add(config_options);

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

    vector<distribution<float> > data;

    // Create data points from -0.8 to 0.8
    for (int i = -800;  i <= 800;  ++i)
        data.push_back(distribution<float>(1, i / 1000.0));

    Auto_Encoder_Stack stack("stack");

    Thread_Context context;

    stack.add(new Twoway_Layer("layer", 1, 1, TF_TANH, MV_DENSE, context));
    
    Auto_Encoder_Trainer trainer;
    trainer.configure("", config);
    trainer.train_stack(stack, data, data, context);

    cerr << stack[0].print() << endl;

    filter_ostream out("data.txt");

#if 0
    for (unsigned i = 0;  i < data.size();  ++i) {
        float output = stack.iapply(stack.apply(data[i]))[0];
        out << format("%6.3f %8.5f %8.5f", data[i][0],
                      output, data[i][0] - output)
            << endl;
    }
#endif
}
