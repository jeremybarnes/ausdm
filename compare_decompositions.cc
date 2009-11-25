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
#include <boost/bind.hpp>

#include "decomposition.h"
#include "svd_decomposition.h"
#include "dnae_decomposition.h"


using namespace std;
using namespace ML;
using namespace ML::DB;

struct Do_Test_Job {

    int x0, x1;

    const SVD_Decomposition & svd;
    const DNAE_Decomposition & dnae1;
    const DNAE_Decomposition & dnae2;
    const DNAE_Decomposition & dnae3;

    Lock & lock;

    double & total_error_svd;
    double & total_error_dnae1;
    double & total_error_dnae2;
    double & total_error_dnae3;

    const Data & data;

    int order_svd, order_dnae;

    vector<distribution<float> > & decomp_svd;
    vector<distribution<float> > & decomp_dnae1;
    vector<distribution<float> > & decomp_dnae2;
    vector<distribution<float> > & decomp_dnae3;

    Do_Test_Job(int x0, int x1,
                const SVD_Decomposition & svd,
                const DNAE_Decomposition & dnae1,
                const DNAE_Decomposition & dnae2,
                const DNAE_Decomposition & dnae3,
                Lock & lock,
                double & total_error_svd,
                double & total_error_dnae1,
                double & total_error_dnae2,
                double & total_error_dnae3,
                const Data & data,
                int order_svd, int order_dnae,
                vector<distribution<float> > & decomp_svd,
                vector<distribution<float> > & decomp_dnae1,
                vector<distribution<float> > & decomp_dnae2,
                vector<distribution<float> > & decomp_dnae3)
        : x0(x0), x1(x1), svd(svd), dnae1(dnae1), dnae2(dnae2), dnae3(dnae3),
          lock(lock),
          total_error_svd(total_error_svd),
          total_error_dnae1(total_error_dnae1),
          total_error_dnae2(total_error_dnae2),
          total_error_dnae3(total_error_dnae3),
          data(data), order_svd(order_svd), order_dnae(order_dnae),
          decomp_svd(decomp_svd),
          decomp_dnae1(decomp_dnae1),
          decomp_dnae2(decomp_dnae2),
          decomp_dnae3(decomp_dnae3)
    {
    }

    void operator () ()
    {
        double l_total_error_svd = 0.0;
        double l_total_error_dnae1 = 0.0;
        double l_total_error_dnae2 = 0.0;
        double l_total_error_dnae3 = 0.0;

        for (unsigned x = x0;  x < x1;  ++x) {

            distribution<float> input = data.examples[x]->models;

            distribution<float> out_svd = svd.decompose(input);
            distribution<float> out_dnae1 = dnae1.decompose(input);
            distribution<float> out_dnae2 = dnae2.decompose(input);
            distribution<float> out_dnae3 = dnae3.decompose(input);

            decomp_svd[x] = out_svd;
            decomp_dnae1[x] = out_dnae1;
            decomp_dnae2[x] = out_dnae2;
            decomp_dnae3[x] = out_dnae3;
            decomp_dnae4[x] = out_dnae4;

            distribution<float> recomp_svd = svd.recompose(input, out_svd, order_svd);
            distribution<float> recomp_dnae1 = dnae1.recompose(input, out_dnae1, order_dnae);
            distribution<float> recomp_dnae2 = dnae2.recompose(input, out_dnae2, order_dnae);
            distribution<float> recomp_dnae3 = dnae3.recompose(input, out_dnae3, order_dnae);
            distribution<float> err_svd = input - recomp_svd;
            distribution<float> err_dnae1 = input - recomp_dnae1;
            distribution<float> err_dnae2 = input - recomp_dnae2;
            distribution<float> err_dnae3 = input - recomp_dnae3;
            
            l_total_error_svd += err_svd.dotprod(err_svd);
            l_total_error_dnae1 += err_dnae1.dotprod(err_dnae1);
            l_total_error_dnae2 += err_dnae2.dotprod(err_dnae2);
            l_total_error_dnae3 += err_dnae3.dotprod(err_dnae3);
        }

        Guard guard(lock);

        total_error_svd += l_total_error_svd;
        total_error_dnae1 += l_total_error_dnae1;
        total_error_dnae2 += l_total_error_dnae2;
        total_error_dnae3 += l_total_error_dnae3;
    }
};

int main(int argc, char ** argv)
{
    // What type of target do we predict?
    string target_type;

    // Which size (S, M, L for Small, Medium and Large)
    string size = "S";

    {
        using namespace boost::program_options;

        options_description control_options("Control Options");

        control_options.add_options()
            ("target-type,t", value<string>(&target_type),
             "select target type: auc or rmse")
            ("size,S", value<string>(&size),
             "size: S (small), M (medium) or L (large)");

        options_description all_opt;
        all_opt
            .add(control_options);

        all_opt.add_options()
            ("help,h", "print this message");
        
        variables_map vm;
        store(command_line_parser(argc, argv)
              .options(all_opt)
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

    string targ_type_uc;
    if (target == AUC) targ_type_uc = "AUC";
    else if (target == RMSE) targ_type_uc = "RMSE";
    else throw Exception("unknown target type");

    Data data_train;
    data_train.load("download/" + size + "_" + targ_type_uc + "_Train.csv.gz",
                    target);
    
    
    string decomp1 = "loadbuild/" + size + "_" + target_type + "_SVD.dat";
    string decomp2 = "loadbuild/" + size + "_" + target_type + "_DNAE.dat";
    string decomp3 = "loadbuild/" + size + "_" + target_type + "_DNAE2.dat";
    string decomp4 = "loadbuild/" + size + "_" + target_type + "_DNAE3.dat";

    boost::shared_ptr<SVD_Decomposition> svd
        = boost::dynamic_pointer_cast<SVD_Decomposition>
        (Decomposition::load(decomp1));

    boost::shared_ptr<DNAE_Decomposition> dnae1
        = boost::dynamic_pointer_cast<DNAE_Decomposition>
        (Decomposition::load(decomp2));

    boost::shared_ptr<DNAE_Decomposition> dnae2
        = boost::dynamic_pointer_cast<DNAE_Decomposition>
        (Decomposition::load(decomp3));

    boost::shared_ptr<DNAE_Decomposition> dnae3
        = boost::dynamic_pointer_cast<DNAE_Decomposition>
        (Decomposition::load(decomp4));

    int nx = data_train.nx();

    for (unsigned i = 0;  i < dnae1->stack.size();  ++i) {

        int order = dnae1->stack[i].outputs();

        svd->extract_for_order(order);

        cerr << "i = " << i << " order = " << order << endl;

        double total_error_svd = 0.0,
            total_error_dnae1 = 0.0,
            total_error_dnae2 = 0.0,
            total_error_dnae3 = 0.0;

        static Worker_Task & worker = Worker_Task::instance(num_threads() - 1);
        
        Lock lock;

        vector<distribution<float> > decomp_svd(nx);
        vector<distribution<float> > decomp_dnae1(nx);
        vector<distribution<float> > decomp_dnae2(nx);
        vector<distribution<float> > decomp_dnae3(nx);
        
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
            
            for (unsigned x = 0;  x < nx;  x += 100, ++job_num) {
                int last = std::min<int>(nx, x + 100);
                
                // Create the job
                Do_Test_Job job(x, last,
                                *svd, *dnae1, *dnae2, *dnae3,
                                lock, total_error_svd, total_error_dnae1,
                                total_error_dnae2, total_error_dnae3,
                                data_train, order, i,
                                decomp_svd, decomp_dnae1, decomp_dnae2,
                                decomp_dnae3);
                // Send it to a thread to be processed
                worker.add(job, "blend job", group);
            }
        }
        
        // Add this thread to the thread pool until we're ready
        worker.run_until_finished(group);

        double err_svd = sqrt(total_error_svd / nx);
        double err_dnae1 = sqrt(total_error_dnae1 / nx);
        double err_dnae2 = sqrt(total_error_dnae2 / nx);
        double err_dnae3 = sqrt(total_error_dnae3 / nx);
        
        cerr << "RMSE: svd " << err_svd << " dnae1 " << err_dnae1
             << " dnae2 " << err_dnae2 << " dnae3 " << err_dnae3 << endl;
    }
}
