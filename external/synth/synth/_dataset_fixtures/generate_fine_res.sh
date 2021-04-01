rm -rf fine_res_budget
mkdir fine_res_budget

for time in 20160801.0022300 20160801.0037300
do
    for tile in {1..6}
    do
        ln -s ../fine_res_budget.json fine_res_budget/${time}.tile${tile}.nc.json
    done
done