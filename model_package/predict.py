'''
  Run this file to get prediction.
  @python version : 3.6.8
  @author : pangjc
  @time : 2019/12/25
'''

import csv
import numpy as np
import lightgbm as lgb

model_num = 10

def get_predict_result():

    bst_list = []
    for i in range(model_num):
        bst_list.append(lgb.Booster(model_file='./model_file/lgb_model%d.bin'%i))

    weight = np.load('./model_file/weight.npy').tolist()

    missed_num = 0

    test_dataset = {}
    with open('./data/test_data.csv', 'r') as f:
        data = csv.reader(f)
        for row in data:
            if row[0] == "build_id":
                continue

            test_dataset[row[0]] = row[1:]

    with open('./data/Non_errored_build_ids.csv', 'r') as f:
        data = csv.reader(f)

        with open("./pred_result.csv", "w") as result:
            writer = csv.writer(result)
            writer.writerow(["ids", "prediction"])

            key_list = test_dataset.keys()
            for row in data:
                if row[0] == "ids":
                    continue

                this_build_id = row[0]
                if this_build_id in key_list:
                    pred_sum = 0
                    this_test_data = np.array([test_dataset[this_build_id]])

                    for i in range(model_num):
                        pred_sum += bst_list[i].predict(this_test_data) * weight[i]

                    if pred_sum >= model_num * 0.5:
                        writer.writerow([row[0], 1])
                    else:
                        writer.writerow([row[0], 0])

                else:
                    missed_num += 1
                    writer.writerow([row[0], 0])

    print("%d keys have been missed" % missed_num)

if __name__ == "__main__":
    get_predict_result()
