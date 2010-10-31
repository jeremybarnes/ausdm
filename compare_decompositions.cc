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
#include <boost/bind.hpp>

#include "decomposition.h"
#include "svd_decomposition.h"
#include "dnae_decomposition.h"
#include "jml/stats/distribution_simd.h"
#include "jml/stats/distribution_ops.h"
#include "jml/algebra/lapack.h"


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

            distribution<float> out_svd = svd.decompose(input, order_svd);
            distribution<float> out_dnae1 = dnae1.decompose(input, order_dnae);
            distribution<float> out_dnae2 = dnae2.decompose(input, order_dnae);
            distribution<float> out_dnae3 = dnae3.decompose(input, order_dnae);

            decomp_svd[x] = out_svd;
            decomp_dnae1[x] = out_dnae1;
            decomp_dnae2[x] = out_dnae2;
            decomp_dnae3[x] = out_dnae3;

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

distribution<double>
calc_r_squared(distribution<float> & labels,
               double label_mean, double label_std,
               const vector<distribution<float> > & outputs)
{
    if (outputs.empty())
        throw Exception("outputs are empty");

    int nx = outputs.size(), nm = outputs[0].size();

    if (nm == 0) throw Exception("no models");
    if (labels.size() != nx)
        throw Exception("no labels");


    int nx_svd = std::min(5000, nx);
    int nvalues = std::min(nx_svd, nm);

    distribution<double> svalues(nvalues);
    boost::multi_array<double, 2> data(boost::extents[nx_svd][nm]);

    for (unsigned x = 0;  x < nx_svd;  ++x) {
        for (unsigned m = 0;  m < nm;  ++m)
            data[x][m] = outputs[x][m];
    }
    
    
    int result = LAPack::gesdd("N", nm, nx_svd,
                               data.data(), nm,
                               &svalues[0],
                               0, nm,
                               0, nvalues);
    if (result != 0)
        throw Exception("gesdd returned non-zero");

    //cerr << "svalues = " << svalues << endl;
    //int nsvalues = (svalues >= svalues.max() * 0.01).count();


    distribution<double> output_totals(nm);

    for (unsigned x = 0;  x < nx;  ++x)
        output_totals += outputs[x];

    distribution<double> output_means = output_totals / nx;

    distribution<double> output_total_variance(nm);
    distribution<double> output_correlation(nm);

    for (unsigned x = 0;  x < nx;  ++x) {
        output_total_variance += sqr(outputs[x] - output_means);
        output_correlation
            += (outputs[x] - output_means)
            * (labels[x] - label_mean);
    }

    distribution<double> stds = sqrt(output_total_variance / nx);

    //cerr << "stds = " << stds << endl;
   
    return output_correlation / (nx * label_std * stds);
}


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

    Data data;
    data.load("download/" + size + "_" + targ_type_uc + "_Train.csv.gz",
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

    int nx = data.nx();

    double label_mean = data.targets.mean();
    double label_std  = data.targets.std();

    //cerr << "label_mean = " << label_mean << endl;
    //cerr << "label_std = " << label_std << endl;

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
                                data, order, i,
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

        // Now calculate the r^2 for each of the features
        distribution<double> r2_svd
            = calc_r_squared(data.targets, label_mean, label_std,
                             decomp_svd);
        r2_svd = abs(r2_svd);
        std::sort(r2_svd.rbegin(), r2_svd.rend());
        cerr << "SVD: highest " << r2_svd[0] << " 10th: " << r2_svd[9]
             << " mean: " << r2_svd.mean() << endl;


        distribution<double> r2_dnae1
            = calc_r_squared(data.targets, label_mean, label_std,
                             decomp_dnae1);
        r2_dnae1 = abs(r2_dnae1);
        std::sort(r2_dnae1.rbegin(), r2_dnae1.rend());
        cerr << "DNAE1: highest " << r2_dnae1[0] << " 10th: " << r2_dnae1[9]
             << " mean: " << r2_dnae1.mean() << endl;
        

        distribution<double> r2_dnae2
            = calc_r_squared(data.targets, label_mean, label_std,
                             decomp_dnae2);
        r2_dnae2 = abs(r2_dnae2);
        std::sort(r2_dnae2.rbegin(), r2_dnae2.rend());
        cerr << "DNAE2: highest " << r2_dnae2[0] << " 10th: " << r2_dnae2[9]
             << " mean: " << r2_dnae2.mean() << endl;
        
        distribution<double> r2_dnae3
            = calc_r_squared(data.targets, label_mean, label_std,
                             decomp_dnae3);
        r2_dnae3 = abs(r2_dnae3);
        std::sort(r2_dnae3.rbegin(), r2_dnae3.rend());
        cerr << "DNAE3: highest " << r2_dnae3[0] << " 10th: " << r2_dnae3[9]
             << " mean: " << r2_dnae3.mean() << endl;
        
    }
}
