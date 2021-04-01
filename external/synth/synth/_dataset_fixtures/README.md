To update the nudge to fine fixtures run 

    rm -rf nudge_to_fine/
    synth-read-schema gs://vcm-ml-scratch/test-end-to-end-integration/integration-test-cf6e957cea82/nudge_to_fine_run/ nudge_to_fine/

Then change the dimensions to match the fine res data

    sed -i.bak 's/79/19/g' *.json
    sed -i.bak 's/4/2/g' *.json
    sed -i.bak 's/49/9/g' *.json
    sed -i.bak 's/48/8/g' *.json

Finally, open the files manually and delete entries from the time coordinate so
it actually has size 2.

