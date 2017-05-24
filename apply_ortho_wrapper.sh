# conditions = ["classic", "plan", "interprut"]

for cond in "classic" "plan" "interprut"
do
    for sub in $(<subjects.txt)
    do
        fname="$sub""_""$cond""_ts""_DKT""_snr-3-epo.mat"
        outname="$sub""_""$cond""_ts""_DKT""_snr-3_orth-epo.mat"
        echo "running sub: $sub"
        /Applications/MATLAB_R2016b.app/bin/matlab -nojvm -nodisplay -nosplash -r "apply_orthoganlisation('$fname')"; exit;
        # eval submit_to_cluster -q $2 \"python $1 $sub\"
        mv data_org.mat $outname
    done
done
