import cv2
import glob
import pandas as pd

emp_dict = {}

emp_dict["filename"] = []
emp_dict["sex"] = []
emp_dict["age"] = []
emp_dict["condition"] = []

file_path_monty = "/home/rosguy/PDC_Paper_Dataset/MontgomerySet/ClinicalReadings/*"
file_path_china = "/home/rosguy/PDC_Paper_Dataset/MontgomerySet/ChinaSet_AllFiles/ClinicalReadings/*"

china = 1
count = 0
for filename in glob.glob(file_path_china):
    
    f_name = filename.split("/")[-1]
    print(f_name) 
    print(count)
    with open(filename,"r+") as f:
        lines = f.readlines()
        print(lines[1])
    
        if(china==1):
            sex = lines[0].split(" ")[0]
            condition = lines[1]
            
            if(lines[0].split()[1]==","):
                if('yrs' in lines[0]):
                    age = (int)(lines[0].split(" ")[2].rstrip('yrs\n'))
                elif('month' in lines[0]):
                    age = (int)(lines[0].split(" ")[2].rstrip('month\n'))
                elif('days' in lines[0]):
                    age = (int)(lines[0].split(" ")[2].rstrip('days\n'))
                    
                
            else:
                if('yrs' in lines[0]):
                    age = (int)(lines[0].split()[1].rstrip('yrs\n'))
                elif('month' in lines[0]):
                    age = (int)(lines[0].split()[1].rstrip('month\n'))
                elif('days' in lines[0]):
                    age = (int)(lines[0].split()[1].rstrip('days\n'))           
        else:
            sex = lines[0].split(":")[1].rstrip("\n")
            age = (int)(lines[1].split(":")[1].lstrip(" 0").rstrip("Y\n"))
            condition = lines[2].rstrip("\n")
            
            
            #print(lines[1].split(":")[1].lstrip(" 0").rstrip("Y\n"))
            #print(age)
            #print(condition)
        emp_dict["filename"].append(f_name)
        emp_dict["sex"].append(sex)
        emp_dict["age"].append(age)
        emp_dict["condition"].append(condition)
        count+=1
    
df = pd.DataFrame.from_dict(emp_dict)

df.to_csv("data_china.csv",index=False)

