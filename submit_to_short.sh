
subjects = ["0008", "0009", "0010", "0012", "0013",
            "0014", "0015","0016", "0017", "0018",  "0019", "0020",
            "0021", "0022"]

for sub in "0008"  #"0009" "0010" "0012" "0013" "0014" "0015" "0016" "0017" "0018" "0019" "0020" "0021" "0022"
do
    qsub -j y -q short.q hilbert_transform_ts.py $sub
done
