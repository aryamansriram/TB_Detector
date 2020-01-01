import cv2
import glob
import matplotlib.pyplot as plt
import numpy as np
import pickle



count = 0


def masker(img,filename):
    l_mask = cv2.imread("/home/rosguy/PDC_Paper_Dataset/MontgomerySet/ManualMask/leftMask/"+filename,0)
    l_mask = cv2.resize(l_mask,(512,512))

    r_mask = cv2.imread("/home/rosguy/PDC_Paper_Dataset/MontgomerySet/ManualMask/rightMask/"+filename,0)
    r_mask = cv2.resize(r_mask,(512,512))
    
    
    masked_left = cv2.bitwise_and(img,img,mask=l_mask)

    masked_right = cv2.bitwise_and(img,img,mask=r_mask)
    
    return masked_left,masked_right
    
#masker(masked_left,"MCUCXR_0099_0.png")  

def cropper(masked_img):
    flag_r = 0
    flag_l = 0
    flag_t = 0
    flag_b = 0
    for i in range(511,0,-1):
        vec = masked_img[:,i,2]
        if(flag_r==0):  
            
            vec = masked_img[:,i,2]
            max_index = np.argmax(vec)
            max_val_r = vec[max_index]
            if(max_val_r>100):
                
                r_crop_index = i
                #print("R_INDEX",r_crop_index)
                flag_r=1
            
        if(flag_l==0):
            
            vec = masked_img[:,512-i,2]
            max_index = np.argmax(vec)
            max_val_l = vec[max_index]
        
            if(max_val_l>100):
                l_crop_index = 512-i
                #print("L INDEX",l_crop_index)
                flag_l=1
                
        if(flag_t==0):
           
            vec = masked_img[512-i,:,2]
            max_index = np.argmax(vec)
            max_val_t = vec[max_index]

            if(max_val_t>100):
                t_crop_index = 512-i
                #print("Top_Index",t_crop_index)
                flag_t=1
        
        if(flag_b==0):
            
            vec = masked_img[i,:,2]
            max_index = np.argmax(vec)
            max_val_b = vec[max_index]

            if(max_val_b>100):
                b_crop_index = i
                #print("bottom Index",b_crop_index)
                flag_b=1
                
        if(flag_t==1 and flag_r==1 and flag_l==1 and flag_b==1):
            break
    
    return masked_img[t_crop_index:b_crop_index,l_crop_index:r_crop_index,:]




'''masked_left1,masked_right1 = masker(img,"MCUCXR_0099_0.png")  

cropped_left = cropper(masked_left1)
cropped_right = cropper(masked_right1)'''

img_arr = np.zeros((138,224,448,3),dtype="uint8")
count=0

for filename in glob.glob("/home/rosguy/PDC_Paper_Dataset/MontgomerySet/CXR_png/*"):
    img = cv2.imread(filename)
    f_name = filename.split("/")[-1]

    if(filename=="/home/rosguy/PDC_Paper_Dataset/MontgomerySet/CXR_png/Thumbs.db"):
        continue
    
    img = cv2.resize(img,(512,512))
    masked_left1,masked_right1 = masker(img,f_name)  

    cropped_left = cropper(masked_left1)
    cropped_right = cropper(masked_right1)
  
  
    cropped_left = cv2.resize(cropped_left,(224,224))
    cropped_right = cv2.resize(cropped_right,(224,224))


    img_fin = np.concatenate((cropped_left,cropped_right),axis=1)

    print(img_fin.shape)    
    img_arr[count,:,:,:] = img_fin
    
    
    
          
    count+=1
    


file = open('img_array.pkl','wb')
pickle.dump(img_arr,file)
file.close()





'''
l_mask = cv2.imread("/home/rosguy/PDC_Paper_Dataset/MontgomerySet/ManualMask/leftMask/MCUCXR_0099_0.png",0)
l_mask = cv2.resize(l_mask,(512,512))

r_mask = cv2.imread("/home/rosguy/PDC_Paper_Dataset/MontgomerySet/ManualMask/rightMask/MCUCXR_0099_0.png",0)
r_mask = cv2.resize(r_mask,(512,512))


masked_left = cv2.bitwise_and(img,img,mask=l_mask)

masked_right = cv2.bitwise_and(img,img,mask=r_mask)

vis = np.concatenate((masked_left, masked_right), axis=0)

max_px = 0
max_index = 0

arr = masked_left[:,:,2]

for i in range(511,0,-1):
    max_val = masked_left[512-i,:,2][np.argmax(masked_left[512-i,:,2])]
    print(i)
    if(max_val>100):
        crop_index = i
        print(i)
        print(crop_index)
        break



        

plt.imshow(masked_left[:crop_index,:,:])
plt.imshow(masked_left)
'''
#############################  CHINA  ####################################
import glob
import cv2

data_china = pd.read_csv("data_china.csv")

labels = []
count = 0
img_arr = np.zeros((138,224,224,3))
for filename in glob.glob("/home/rosguy/PDC_Paper_Dataset/ChinaSet_AllFiles/MontgomerySet/CXR_png/*"):
    if(filename=="/home/rosguy/PDC_Paper_Dataset/ChinaSet_AllFiles/MontgomerySet/CXR_png/Thumbs.db"):
        continue    
    img = cv2.imread(filename)
    img = cv2.resize(img,(512,512))
    img_arr[count,:,:,:] = img
    count+=1
    f_name = filename.split("/")[-1]
    f_name = f_name.split(".")[0]+".txt"
    ser = data[data["filename"]==f_name]
    labels.append(ser["condition"].values[0])

img_arr = img_arr.astype(np.uint8)

file = open('img_array_china.pkl','wb')
pickle.dump(img_arr,file)
file.close()




data_monty = pd.read_csv("data.csv")

labels = []
count = 0
img_arr = np.zeros((138,512,512,3))
for filename in glob.glob("/home/rosguy/PDC_Paper_Dataset/MontgomerySet/CXR_png/*"):
    if(filename=="/home/rosguy/PDC_Paper_Dataset/MontgomerySet/CXR_png/Thumbs.db"):
        continue    
    img = cv2.imread(filename)
    img = cv2.resize(img,(512,512))
    img_arr[count,:,:,:] = img
    count+=1
    f_name = filename.split("/")[-1]
    f_name = f_name.split(".")[0]+".txt"
    ser = data[data["filename"]==f_name]
    labels.append(ser["condition"].values[0])

img_arr = img_arr.astype(np.uint8)

file = open('img_array_china.pkl','wb')
pickle.dump(img_arr,file)
file.close()





    
#plt.imshow(img_fin)