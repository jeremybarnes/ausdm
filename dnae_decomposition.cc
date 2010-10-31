/* dnae_decomposition.cc
   Jeremy Barnes, 24 October 2009
   Copyright (c) 2009 Jeremy Barnes.  All rights reserved.

   Denoising autoencoder decomposition.
*/

#include "dnae_decomposition.h"
#include "svd_decomposition.h"

#include "jml/algebra/matrix_ops.h"
#include "jml/math/xdiv.h"
#include "jml/algebra/lapack.h"
#include "jml/arch/atomic_ops.h"

#include <boost/progress.hpp>
#include <boost/bind.hpp>
#include <boost/tuple/tuple.hpp>
#include <boost/assign/list_of.hpp>

#include "jml/utils/worker_task.h"
#include "jml/boosting/registry.h"

#include "jml/arch/timers.h"
#include "jml/utils/info.h"
#include "jml/utils/guard.h"
#include "jml/arch/threads.h"
#include "jml/arch/atomic_ops.h"
#include "jml/stats/distribution_simd.h"
#include "jml/stats/distribution_ops.h"

#include "jml/db/persistent.h"

#include <limits>

#include "jml/neural/auto_encoder_trainer.h"
#include "jml/neural/twoway_layer.h"


using namespace std;
using namespace ML;
using namespace ML::DB;


namespace ML {

namespace {
static const float NaN = numeric_limits<float>::quiet_NaN();
} // file scope

void calc_W_updates(double k1, const double * x, double k2, const double * y,
                    const double * z, double * r, size_t n)
{
    return SIMD::vec_k1_x_plus_k2_y_z(k1, x, k2, y, z, r, n);
}

void calc_W_updates(float k1, const float * x, float k2, const float * y,
                    const float * z, float * r, size_t n)
{
    return SIMD::vec_k1_x_plus_k2_y_z(k1, x, k2, y, z, r, n);
}

} // namespace ML


/*****************************************************************************/
/* DNAE_DECOMPOSITION                                                        */
/*****************************************************************************/

DNAE_Decomposition::
DNAE_Decomposition()
{
}

distribution<float>
DNAE_Decomposition::
decompose(const distribution<float> & model_outputs, int order) const
{
    if (order == -1) order = stack.size() - 1;
    if (order > stack.size() - 1)
        order = stack.size() - 1;

    distribution<float> output = 0.8 * (model_outputs - means), result;
    
    // Go down the stack
    for (unsigned l = 0;  l <= order;  ++l) {
        output = stack[l].apply(output);
        if (l == order)
            result.insert(result.begin(), output.begin(), output.end());
    }
    
    return result;

    //return stack.apply(0.8 * model_outputs);
}

distribution<float>
DNAE_Decomposition::
recompose(const distribution<float> & model_outputs,
          const distribution<float> & decomposition, int order) const
{
    distribution<float> output = 0.8 * (model_outputs - means);

    // Go down the stack
    int l;
    for (l = 0;  l < stack.size() && l <= order;  ++l) {
        output = stack[l].apply(output);
    }

    // Go the other way and re-generate
    for (; l > 0;  --l) {
        output = stack[l - 1].iapply(output);
    }
    
    return (1.25 * output) + means;
}

std::vector<int>
DNAE_Decomposition::
recomposition_orders() const
{
    vector<int> result;
    for (unsigned i = 0;  i < stack.size();  ++i)
        result.push_back(i);
    return result;
}

void
DNAE_Decomposition::
serialize(ML::DB::Store_Writer & store) const
{
    store << (char)2; // version
    stack.serialize(store);
    store << means;
}

void
DNAE_Decomposition::
reconstitute(ML::DB::Store_Reader & store)
{
    char version;
    store >> version;
    if (version == 1) {
        stack.reconstitute(store);
        means.clear();
    }
    else if (version == 2) {
        stack.reconstitute(store);
        store >> means;
    }
    else throw Exception("DNAE_Decomposition: version was wrong");

}

std::string
DNAE_Decomposition::
class_id() const
{
    return "DNAE";
}

void
DNAE_Decomposition::
train(const Data & training_data,
      const Data & testing_data,
      const Configuration & config)
{
    Thread_Context thread_context;

    int nx = training_data.nx();
    int nxt = testing_data.nx();

    vector<distribution<float> > layer_train(nx), layer_test(nxt);

    // Condition by removing the mean and using unit standard deviation
    means.clear();
    means.resize(training_data.nm());

    for (unsigned x = 0;  x < nx;  ++x)
        means += training_data.examples[x]->models;

    means /= nx;

    distribution<double> stds(means.size());
    for (unsigned x = 0;  x < nx;  ++x) {
        stds += sqr(training_data.examples[x]->models - means) / nx;
    }

    stds = sqrt(stds);

    //cerr << "means = " << means << endl;
    //cerr << "stds  = " << stds << endl;

#if 0 // temporary to fix old version
    boost::shared_ptr<Decomposition> loaded
        = load("auc_decomposed_10.bin");

    DNAE_Decomposition & other
        = dynamic_cast<DNAE_Decomposition &>(*loaded);

    cerr << "successfully loaded old model" << endl;

    stack = other.stack;

    return;
#endif // temporary code
    

    for (unsigned x = 0;  x < nx;  ++x) {
        //layer_train[x] = 0.8f * training_data.examples[x];
        //layer_train[x] = (training_data.examples[x] - means) * (0.8 / stds);
        layer_train[x] = (training_data.examples[x]->models - means) * (0.8);
    }

    for (unsigned x = 0;  x < nxt;  ++x) {
        //layer_test[x] = 0.8f * testing_data.examples[x];
        //layer_test[x] = (testing_data.examples[x] - means) * (0.8 / stds);
        layer_test[x] = (testing_data.examples[x]->models - means) * (0.8);
    }

    vector<int> layer_sizes
        = boost::assign::list_of<int>(250)(150)(100)(50);

    config.get(layer_sizes, "layer_sizes");

    Transfer_Function_Type transfer_function = TF_TANH;

    config.get(transfer_function, "transfer_function");

    stack.clear();

    for (unsigned i = 0;  i < layer_sizes.size();  ++i) {
        int ni = (i == 0 ? layer_train[0].size() : layer_sizes[i - 1]);
        int no = layer_sizes[i];

        stack.add(new Twoway_Layer(format("dnae%d", i),
                                   ni, no, transfer_function,
                                   MV_DENSE, thread_context));
    }


    Auto_Encoder_Trainer trainer;
    trainer.configure("", config);
    trainer.train_stack(stack, layer_train, layer_test, thread_context);
}

void
DNAE_Decomposition::
init(const Configuration & config)
{
}

namespace {

Register_Factory<Decomposition, DNAE_Decomposition>
    DNAE_REGISTER("DNAE");

} // file scope

