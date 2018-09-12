awk '{for(i=9;i<NF;i++){printf "%s ", $i}; printf "%s\n",$NF}' semeval.txt

