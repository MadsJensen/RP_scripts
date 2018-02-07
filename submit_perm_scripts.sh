submit_to_cluster "python perm_test_source_st_FS.py Alpha" -q highmem.q -n 1
submit_to_cluster "python perm_test_source_st_FS.py Beta" -q highmem.q -n 1
submit_to_cluster "python perm_test_source_st_FS.py Gamma_low" -q highmem.q -n 1
submit_to_cluster "python perm_test_source_st_FS.py Gamma_high_1" -q highmem.q -n 1
submit_to_cluster "python perm_test_source_st_FS.py Gamma_high_2" -q highmem.q -n 1
