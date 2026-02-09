from os import listdir
from os.path import isdir, join
import os

global_path = "./result_until_1312026/"

best_macro = -1.0
current_path = ""

for x in listdir(global_path):
    sub_dir = join(global_path, x)

    if not isdir(sub_dir) or '.' not in x:
        continue

    # duyệt tất cả file trong sub_dir
    for fname in listdir(sub_dir):
        if fname.startswith("test_results") and fname.endswith(".txt"):
            file_path = join(sub_dir, fname)

            with open(file_path, "r") as f:
                lines = [l.strip() for l in f.readlines()]

            # giả sử macro nằm ở dòng đầu
            macro_line = lines[0]
            macro_score = float(macro_line.split("=")[1].strip())

            if macro_score > best_macro:
                best_macro = macro_score
                current_path = sub_dir

print("Best run:", current_path)
print("Best macro:", best_macro)
