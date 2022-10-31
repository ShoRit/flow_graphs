# Commands to run the code

python3 pred_ee.py --data chemu_ee --name [MODEL_NAME] --crf --long 25

python3 pred_ee.py --data chemu_ee --name  bert-crf_07_10_2022_20:00:32_b3aa --crf --long 25 --retrain --restore

[MODEL_NAME] will be saved in models/main/ and corresponding log will appear as well in logs



