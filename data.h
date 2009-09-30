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

struct Data {
    void load(const std::string & filename);

    std::vector<float> targets;
    std::vector<std::string> model_names;
    std::vector<int> model_ids;

    struct Model : public std::vector<float> {
    };
    std::vector<Model> models;
};

#endif /* __ausdm__data_h__ */


