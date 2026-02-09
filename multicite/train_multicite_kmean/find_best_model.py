from os import listdir
from os.path import isfile , isdir
import os 
global_path = "."

list_dir  = listdir(global_path)
best_strict = 0 
best_weak = 0
current_path = ''
for x in list_dir: 
    sub_dir  = os.path.join(global_path , x)
    if isdir(sub_dir) and '.' in x:
        if os.path.exists(os.path.join(sub_dir, 'global_step_best.txt')):
            global_step_best_path = os.path.join(sub_dir, 'global_step_best.txt')
            with open(global_step_best_path , 'r') as file :
                best_step = file.read().strip()
            test_result_path = os.path.join(sub_dir, 'test_results'+best_step+'.txt')
            try:
                with open(test_result_path , 'r') as file:
                    result_lines =  [t.strip() for t in  file.readlines() ] 
                    strict_line = result_lines[1]
                    strict_score = float(strict_line.split('=')[1].strip())

                    weak_line = result_lines[2]
                    weak_score = float(weak_line.split('=')[1].strip())

                    if round(strict_score , 2) >= best_strict and round(weak_score , 2)  >= best_weak:
                        best_strict = round(strict_score , 2) 
                        best_weak = round(weak_score , 2)
            except:
                pass
print(best_weak)
print(current_path)
