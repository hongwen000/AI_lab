awk '{for(i=2;i<8;i++){printf "%s ", $i}; printf "%s\n",$8}' semeval.txt | sed  "s/[a-z:]//g"

