/* svd_decomposition.cc
   Jeremy Barnes, 24 October 2009
   Copyright (c) 2009 Jeremy Barnes.  All rights reserved.

   Decomposition using the SVD.
*/

#include "svd_decomposition.h"
#include "boosting/registry.h"
#include "algebra/lapack.h"
#include "stats/distribution_simd.h"
#include "arch/simd_vector.h"


using namespace std;
using namespace ML;
using namespace ML::DB;


/*****************************************************************************/
/* SVD_DECOMPOSITION                                                         */
/*****************************************************************************/

SVD_Decomposition::
SVD_Decomposition()
    : order(0), nm(0), nx(0)
{
}

void
SVD_Decomposition::
train(const Data & training_data,
      const Data & testing_data,
      const ML::Configuration & config)
{
    int order = -1;
    config.find(order, "order");

    if (training_data.examples.empty()) return;

    train(training_data.examples, order);
}

void
SVD_Decomposition::
train(const std::vector<distribution<float> > & data,
      int order)
{
    nm = data[0].size();
    nx = data.size();
        
    boost::multi_array<float, 2> values(boost::extents[nx][nm]);
        
    for (unsigned j = 0;  j < nx;  ++j)
        for (unsigned i = 0;  i < nm;  ++i)
            values[j][i] = data[j][i];
        
    nvalues = std::min(nm, nx);
        
    //cerr << "nx = " << nx << " nm = " << nm
    //     << " nvalues = " << nvalues << endl;
        
    singular_values.resize(nvalues);
    distribution<float> svalues(nvalues);
    boost::multi_array<float, 2> lvectorsT(boost::extents[nvalues][nm]);
    rvectors.resize(boost::extents[nx][nvalues]);
        

    int result = LAPack::gesdd("S", nm, nx,
                               values.data(), nm,
                               &svalues[0],
                               &lvectorsT[0][0], nm,
                               &rvectors[0][0], nvalues);
        
    if (result != 0)
        throw Exception("gesdd returned non-zero");
        
    // Transpose lvectors
    lvectors.resize(boost::extents[nm][nvalues]);
    for (unsigned i = 0;  i < nm;  ++i)
        for (unsigned j = 0;  j < nvalues;  ++j)
            lvectors[i][j] = lvectorsT[j][i];
        
    int nwanted = nvalues;//std::min(nvalues, 200);
    //nwanted = 50;
        
    singular_values
        = distribution<float>(svalues.begin(), svalues.begin() + nwanted);

    extract_for_order(order);
}

// Set the order of the model and extract things based upon it
void
SVD_Decomposition::
extract_for_order(int order)
{        
    if (order == -1) order = nvalues;

    if (order <= 0)
        throw Exception("invalid order");
    if (order > nm)
        order = nm;

    this->order = order;
    //cerr << "singular_values = " << singular_values << endl;

    singular_models.resize(nm);
    singular_values_order
        = distribution<float>(singular_values.begin(),
                              singular_values.begin() + order);
    
    for (unsigned i = 0;  i < nm;  ++i)
        singular_models[i]
            = distribution<float>(&lvectors[i][0],
                                  &lvectors[i][order - 1] + 1);
}

distribution<float>
SVD_Decomposition::
decompose(const distribution<float> & vals) const
{
    if (singular_values.empty())
        throw Exception("apply_decomposition(): no decomposition was done");
        
    if (vals.size() != nm)
        throw Exception("SVD_Decomposition::decompose(): wrong size");

    // First, get the singular vector for the model
    distribution<double> target_singular(singular_values_order.size());
        
    for (unsigned i = 0;  i < nm;  ++i)
        SIMD::vec_add(&target_singular[0], vals[i],
                      &singular_models[i][0],
                      &target_singular[0], singular_values_order.size());

    target_singular /= singular_values_order;

    //cerr << "singular_values_order = " << singular_values_order << endl;
    //cerr << "singular for model: " << target_singular << endl;
        
    return distribution<float>(target_singular.begin(),
                               target_singular.end());
}

distribution<float>
SVD_Decomposition::
recompose(const distribution<float> & decomposition, int order) const
{
    if (order == -1 || order > decomposition.size())
        order = decomposition.size();

    if (decomposition.size() > this->order)
        throw Exception("unknown decomposition");

    if (singular_values.empty())
        throw Exception("apply_decomposition(): no decomposition was done");
        
    distribution<float> scaled = decomposition;

    for (unsigned i = 0;  i < order;  ++i)
        scaled[i] *= singular_values_order[i];
    
    distribution<double> result(nm);
    for (unsigned i = 0;  i < nm;  ++i)
        result[i] = SIMD::vec_dotprod_dp(&scaled[0],
                                         &singular_models[i][0],
                                         order);

    return result.cast<float>();
}

void
SVD_Decomposition::
serialize(DB::Store_Writer & store) const
{
    store << string("begin SVD decomposition")
          << (char)1 /* version */
          << compact_size_t(order) << compact_size_t(nm)
          << compact_size_t(nx) << compact_size_t(nvalues);
    store << singular_values << lvectors << rvectors;
    store << string("end SVD decomposition");
}

void
SVD_Decomposition::
reconstitute(DB::Store_Reader & store)
{
    string s;
    store >> s;
    if (s != "begin SVD decomposition")
        throw Exception("expected SVD decomposition");
    char version;
    store >> version;
    if (version == 1) {
        order = compact_size_t(store);
        nm = compact_size_t(store);
        nx = compact_size_t(store);
        nvalues = compact_size_t(store);
        store >> singular_values >> lvectors >> rvectors;
        store >> s;
        if (s != "end SVD decomposition")
            throw Exception("expected end of SVD decomposition");
    }
    else {
        throw Exception("SVD_Decomposition: unknown order");
    }
}

std::string
SVD_Decomposition::
class_id() const
{
    return "SVD";
}

namespace {

Register_Factory<Decomposition, SVD_Decomposition>
    SVD_REGISTER("SVD");

} // file scope

