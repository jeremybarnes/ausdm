/* decomposition.cc                                                -*- C++ -*-
   Jeremy Barnes, 24 October 2009
   Copyright (c) 2009 Jeremy Barnes.  All rights reserved.

   Decomposition code.
*/


#include "decomposition.h"
#include "boosting/registry.h"
#include "utils/filter_streams.h"


using namespace std;
using namespace ML;
using namespace ML::DB;


/*****************************************************************************/
/* DECOMPOSITION                                                             */
/*****************************************************************************/

void
Decomposition::
poly_serialize(ML::DB::Store_Writer & store) const
{
    Registry<Decomposition>::singleton().serialize(store, this);
}

boost::shared_ptr<Decomposition>
Decomposition::
poly_reconstitute(ML::DB::Store_Reader & store)
{
    return Registry<Decomposition>::singleton().reconstitute(store);
}

boost::shared_ptr<Decomposition>
Decomposition::
create(const std::string & type)
{
    return Registry<Decomposition>::singleton().create(type);
}

void
Decomposition::
save(const std::string & filename) const
{
    filter_ostream stream(filename);
    DB::Store_Writer store(stream);
    poly_serialize(store);
}

boost::shared_ptr<Decomposition>
Decomposition::
load(const std::string & filename)
{
    filter_istream stream(filename);
    DB::Store_Reader store(stream);
    return poly_reconstitute(store);
}
