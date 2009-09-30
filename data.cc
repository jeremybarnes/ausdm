/* data.cc
   Jeremy Barnes, 30 September 2009
   Copyright (c) 2009 Jeremy Barnes.  All rights reserved.

   File to read data for the competition.
*/

#include "data.h"
#include "utils/parse_context.h"

using namespace std;
using namespace ML;

void
Data::
load(const std::string & filename)
{
    Parse_Context c(filename);

    // First row: header.
    c.expect_literal("RowID,Target");

    while (!c.match_eol()) {
        c.expect_literal(',');
        string model_name = c.expect_text(",\n");
        model_names.push_back(model_name);
    }

    // Create the data structures
    cerr << model_names.size() << " models... ";
    
    models.resize(model_names.size());
    for (unsigned i = 0;  i < models.size();  ++i) {
        models[i].reserve(50000);
    }
    
    int num_rows = 0;
    for (; c; ++num_rows) {
        int id = c.expect_int();
        model_ids.push_back(id);
        c.expect_literal(',');

        int target = c.expect_int();
        targets.push_back(target);

        for (unsigned i = 0;  i < models.size();  ++i) {
            c.expect_literal(',');
            int score = c.expect_int();
            models[i].push_back(score / 1000.0);
        }

        c.skip_whitespace();
        c.expect_eol();
    }

    cerr << num_rows << " rows... ";
}
