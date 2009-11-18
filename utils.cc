/* utils.cc
   Jeremy Barnes, 2 November 2009
   Copyright (c) 2009 Jeremy Barnes.  All rights reserved.

   Utility functions.
*/

#include "utils.h"
#include "algebra/lapack.h"
#include "stats/distribution_ops.h"

using namespace std;


namespace ML {

template<class Float>
distribution<Float>
perform_irls_impl(const distribution<Float> & correct,
                  const boost::multi_array<Float, 2> & outputs,
                  const distribution<Float> & w,
                  Link_Function link_function,
                  bool ridge_regression)
{
    int nx = correct.size();
    int nv = outputs.shape()[0];

    if (outputs.shape()[1] != nx)
        throw Exception("wrong shape for outputs");

    bool verify = false;
    //verify = true;

    distribution<Float> svalues1
        (std::min(outputs.shape()[0], outputs.shape()[1]) + 1);

    if (verify) {
        boost::multi_array<Float, 2> outputs2 = outputs;

        svalues1.push_back(0.0);

        int result = LAPack::gesdd("N",
                                   outputs2.shape()[1],
                                   outputs2.shape()[0],
                                   outputs2.data(), outputs2.shape()[1], 
                                   &svalues1[0], 0, 1, 0, 1);
    
        if (result != 0)
            throw Exception("error in SVD");
    }
    
    boost::multi_array<Float, 2> outputs3 = outputs;

    /* Factorize the matrix with partial pivoting.  This allows us to find the
       largest number of linearly independent columns possible. */
    
    distribution<Float> tau(nv, 0.0);
    distribution<int> permutations(nv, 0);
    
    int res = LAPack::geqp3(outputs3.shape()[1],
                            outputs3.shape()[0],
                            outputs3.data(), outputs3.shape()[1],
                            &permutations[0],
                            &tau[0]);
    
    if (res != 0)
        throw Exception(format("geqp3: error in parameter %d", -res));

    // Convert back to c indexes
    permutations -= 1;

    //cerr << "permutations = " << permutations << endl;
    //cerr << "tau = " << tau << endl;

    distribution<Float> diag(nv);
    for (unsigned i = 0;  i < nv;  ++i)
        diag[i] = outputs3[i][i];

    //cerr << "diag = " << diag << endl;
    
    int nkeep = nv;
    while (nkeep > 0 && fabs(diag[nkeep - 1]) < 0.01) --nkeep;
    
    //cerr << "keeping " << nkeep << " of " << nv << endl;
    
    vector<int> new_loc(nv, -1);
    for (unsigned i = 0;  i < nkeep;  ++i)
        new_loc[permutations[i]] = i;

    boost::multi_array<Float, 2> outputs_reduced(boost::extents[nkeep][nx]);
    for (unsigned i = 0;  i < nx;  ++i)
        for (unsigned j = 0;  j < nv;  ++j)
            if (new_loc[j] != -1)
                outputs_reduced[new_loc[j]][i] = outputs[j][i];
    
    double svreduced = svalues1[nkeep - 1];

    if (verify && svreduced < 0.001)
        throw Exception("not all linearly dependent columns were removed");

#if 0
    cerr << "v.size() = " << w.size() << endl;
    cerr << "correct.size() = " << correct.size() << endl;
    cerr << "w.total() = " << w.total() << endl;

    cerr << "outputs_reduced: " << outputs_reduced.shape()[0] << "x"
         << outputs_reduced.shape()[1] << endl;
#endif

    distribution<Float> trained;
    if (ridge_regression) {
        Ridge_Regressor regressor(1e-5);
        trained
            = run_irls(correct, outputs_reduced, w, link_function, regressor);
    }
    else {
        Least_Squares_Regressor regressor;
        trained
            = run_irls(correct, outputs_reduced, w, link_function, regressor);
    }

    distribution<Float> parameters(nv);
    for (unsigned v = 0;  v < nv;  ++v)
        if (new_loc[v] != -1)
            parameters[v] = trained[new_loc[v]];
    

    if (abs(parameters).max() > 1000.0) {

        distribution<Float> svalues_reduced
            (std::min(outputs_reduced.shape()[0],
                      outputs_reduced.shape()[1]));
        
#if 0
        filter_ostream out(format("good.model.%d.txt.gz", model));

        out << "model " << model << endl;
        out << "trained " << trained << endl;
        out << "svalues_reduced " << svalues_reduced << endl;
        out << "parameters " << parameters << endl;
        out << "permuations " << permutations << endl;
        out << "svalues1 " << svalues1 << endl;
        out << "correct " << correct << endl;
        out << "outputs_reduced: " << outputs_reduced.shape()[0] << "x"
            << outputs_reduced.shape()[1] << endl;
        
        for (unsigned i = 0;  i < nx;  ++i) {
            out << "example " << i << ": ";
            for (unsigned j = 0;  j < nkeep;  ++j)
                out << " " << outputs_reduced[j][i];
            out << endl;
        }
#endif

        int result = LAPack::gesdd("N", outputs_reduced.shape()[1],
                                   outputs_reduced.shape()[0],
                                   outputs_reduced.data(),
                                   outputs_reduced.shape()[1], 
                                   &svalues_reduced[0], 0, 1, 0, 1);

        //cerr << "model = " << model << endl;
        cerr << "trained = " << trained << endl;
        cerr << "svalues_reduced = " << svalues_reduced << endl;

        cerr << "parameters.two_norm() = " << parameters.two_norm()
             << endl;

        if (result != 0)
            throw Exception("gesdd returned error");
        
        if (svalues_reduced.back() <= 0.001) {
            throw Exception("didn't remove all linearly dependent");
        }

        // We reject this later
        //if (abs(parameters).max() > 1000.0) {
        //    throw Exception("IRLS returned inplausibly high weights");
        //}
        
    }

    //cerr << "irls returned parameters " << parameters << endl;

    return parameters;
}

distribution<float>
perform_irls(const distribution<float> & correct,
             const boost::multi_array<float, 2> & outputs,
             const distribution<float> & w,
             Link_Function link_function,
             bool ridge_regression)
{
    return perform_irls_impl(correct, outputs, w, link_function,
                             ridge_regression);
}

distribution<double>
perform_irls(const distribution<double> & correct,
             const boost::multi_array<double, 2> & outputs,
             const distribution<double> & w,
             Link_Function link_function,
             bool ridge_regression)
{
    return perform_irls_impl(correct, outputs, w, link_function,
                             ridge_regression);
}

} // namespace ML
