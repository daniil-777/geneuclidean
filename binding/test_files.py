import os

path_new = "/Volumes/Ubuntu/data/refined_26.05/"
path_old = "/Volumes/Ubuntu/data/new_refined/"
files_pdb = os.listdir(path_new)
files_pdb_old = os.listdir(path_old)
print("len of a new dataset", len(files_pdb))
print("len of an old dataset", len(files_pdb_old))
files_refined = os.listdir(path_old)
for file in files_pdb_old:
    if (file[0].isdigit() == True):
        if file not in files_refined:
            print("no file", file)
        else:
            sub_files = os.listdir(path_old + file)
            if (len(sub_files) < 4):
                print("less than 4", file)



        
        

     
