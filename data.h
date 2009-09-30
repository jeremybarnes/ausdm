/* data.h                                                          -*- C++ -*-
   Jeremy Barnes, 30 September 2009
   Copyright (c) 2009 Jeremy Barnes.  All rights reserved.

   File to hold the data.
*/

#ifndef __ausdm__data_h__
#define __ausdm__data_h__

#include <boost/multi_array.hpp>
#include <vector>
#include <string>
#include "stats/distribution.h"

/// What kind of target are we calculating?
enum Target {
    RMSE,
    AUC
};

using ML::distribution;


/*****************************************************************************/
/* DATA                                                                      */
/*****************************************************************************/

/** Data structure to contain the dataset that we are working on. */

struct Data {
    void load(const std::string & filename, Target target);

    Target target;

    /// Target values to predict
    distribution<float> targets;

    /// Names of the models
    std::vector<std::string> model_names;

    /// ID values of the models
    std::vector<int> model_ids;

    /// The output of one of the models that we are blending
    struct Model : public distribution<float> {
        
        /// Calculate the RMSE over the given set of targets
        double calc_rmse(const distribution<float> & targets) const;
        
        /// Calculate the AUC over the given set of targets
        double calc_auc(const distribution<float> & targets) const;

        /// Score over whatever target we are trying to calculate
        double score;

        /// Rank over targets we are trying to calculate
        int rank;
    };

    std::vector<Model> models;
};

#endif /* __ausdm__data_h__ */


