
for file in perm_test_classifier*
do
    eval submit_to_cluster -q all.q \"python $file\"
done
