#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy
import os
import shutil
import fileinput


# In[2]:


dir_path = "./darknet"
src_path = os.path.join(dir_path,"src_orig")
bck_path = os.path.join(dir_path,"bck")
mod_path = os.path.join(dir_path,"src3")


# In[3]:


# generate backup files before code modification 
if not os.path.isdir(mod_path):
    os.mkdir(mod_path)
if not os.path.isdir(bck_path) and os.path.isdir(src_path):
    shutil.copytree(src_path, bck_path)


# In[4]:


## RESTORE THE BACKUP FILES
if not os.path.isdir(src_path):
    shutil.copytree(bck_path,src_path)


# In[5]:


# read files & gather device functions information (name, the number of arguments) # find the __global__ functions 
func_info = {}
func_pointer = {}
srcfile_list = [f for f in os.listdir(src_path) if os.path.isfile(os.path.join(src_path, f))]
for srcfile in srcfile_list:
    f = open(os.path.join(src_path,srcfile), 'r')
    codelines = f.readlines()
    for linenum in range(0, len(codelines)):
        current_line = codelines[linenum]
        if "__global__" in current_line:         # gather deivce function lines
            func_lines = [current_line]
            while not ")" in current_line:       # if function defination is more than 1 line
                linenum += 1
                current_line = codelines[linenum]
                func_lines.append(current_line)
            if len(func_lines) != 1:             # make in one line 
                func_line = ""
                for line in func_lines:
                    func_line += line
            else:
                func_line = func_lines[0]
            
            token = func_line.split()
            func_name = token[2].split("(")[0]
            func_info[func_name] = func_line.count(",")+1
            
            tokens = func_line.split(",")
            pointer_bit = []
            for i in range(len(tokens)):
                if "*" in tokens[i]:
                    pointer_bit.append(1)
                else:
                    pointer_bit.append(0)
            
            func_pointer[func_name] = pointer_bit
    f.close()


# In[6]:


# based on info, change func & launch
for srcfile in srcfile_list:
    mod_src_path = os.path.join(mod_path, srcfile)
    f1 = open(os.path.join(src_path, srcfile), "r")
    f2 = open(mod_src_path, "w")
    
    codelines = f1.readlines()
    linenum = 0
    while linenum < len(codelines):
        func_name = ""
        for key in func_info.keys():
            if "__global__" in codelines[linenum]:
                func_name = codelines[linenum].split()[2].split("(")[0]
            if "<<<" in codelines[linenum]:
                func_name = codelines[linenum].split("<<<")[0].split()[0]
            if key == func_name:
                if "<<<" in codelines[linenum]:
                    while not ">>>" in codelines[linenum]:
                        linenum += 1
                    replacement = ">>>({}, ".format(func_info[key])
                    for bit in func_pointer[func_name]:
                        replacement += "{}, ".format(bit)
                    codelines[linenum] = codelines[linenum].replace(">>>(", replacement)
                else:
                    replacement = "(int nargs, "
                    for i in range(len(func_pointer[func_name])):
                        replacement += "int ptr_bit_{}, ".format(i)
                    codelines[linenum] = codelines[linenum].replace("(", replacement)
        linenum += 1
    f1.close()
    for codeline in codelines:
        f2.write(codeline)
    f2.close()


# In[ ]:




