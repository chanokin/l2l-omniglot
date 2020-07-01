#! /bin/bash
source venv3/bin/activate
rm L2L-OMNIGLOT/run_results/data_gen000000000{1,2,3,4,5,6,7,8,9}*
rm L2L-OMNIGLOT/run_results/data_gen00000000{1,2,3,4,5}*
mv L2L-OMNIGLOT/run_results/data_gen0000000000_ind000000000* L2L-OMNIGLOT/mushroom_test/slower_exc/ 
python check_mid_spikes.py 
rm mid_analysis/*; mv *png mid_analysis/
scp -r mid_analysis unix:~/l2l/
scp L2L-OMNIGLOT/mushroom_test/slower_exc/* unix:~/l2l/
