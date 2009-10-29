/* ausdm.cc                                                        -*- C++ -*-
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
#include "utils/info.h"
#include "utils/guard.h"
#include "arch/threads.h"
#include <boost/program_options/cmdline.hpp>
#include <boost/program_options/options_description.hpp>
#include <boost/program_options/positional_options.hpp>
#include <boost/program_options/parsers.hpp>
#include <boost/program_options/variables_map.hpp>

#include "dnae_decomposition.h"


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

    DNAE_Stack stack;
    
    Thread_Context context;

    stack.train_dnae(data, data, config, context);

    cerr << stack[0].print() << endl;

    filter_ostream out("data.txt");

    for (unsigned i = 0;  i < data.size();  ++i) {
        float output = stack.iapply(stack.apply(data[i]))[0];
        out << format("%6.3f %8.5f %8.5f", data[i][0],
                      output, data[i][0] - output)
            << endl;
    }
}
