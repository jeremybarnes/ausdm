BEGIN {
    set = 0;
}

$0 == "next" {
    ++set;
    iter = 0;
    go = 0;
    next;
}

NF == 2 && $1 == "dn1" {
    go=1;
    ++iter;
}

!go {
    next;
}

$1 == "bias" || $1 == "combined" {
    go = 0;
}

iter < 11 {
    models[$1] = 1;
    total[set,$1] += $2;
    scores[set,$1,iter] = $2;
}

iter == 11 {
    models[$1] = 1;
    score[set,$1] = $2;
}

END {
    # Calculate statistics
    for (set = 0;  set < 3;  ++set) {
        for (model in models) {
            mean[set,model] = total[set,model] / 10;
            var[set,model] = 0;
            for (j = 1;  j <= 10;  ++j)
                var[set,model] += (scores[set,model,j] - mean[set,model]) * (scores[set,model,j] - mean[set,model]);
        }
    }

    # Tabulate
    for (model in models) {
        printf("%-10s "\
               "& %6.4f & %5.1f & %5.2f$\\pm$%4.2f"\
               "& %6.4f & %5.1f & %5.2f$\\pm$%4.2f"\
               "& %6.4f & %5.1f & %5.2f$\\pm$%4.2f"\
               " \\\\ \n",
               model,

               score[0,model],
               (baseline1-score[0,model]) * 1000.0,
               mean[0,model],
               sqrt(var[0,model]),

               score[1,model],
               (baseline2-score[1,model]) * 1000.0,
               mean[1,model],
               sqrt(var[1,model]),

               score[2,model],
               (baseline3-score[2,model]) * 1000.0,
               mean[2,model],
               sqrt(var[2,model]));
    }
}