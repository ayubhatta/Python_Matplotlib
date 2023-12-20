#!/usr/bin/env python
# coding: utf-8

# # Matplotlib Functions
# # Sactter Plot

# In[1]:

# importing matplotlib, numpy and pandas
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# In[2]:


plt.style.use('dark_background')


# In[29]:


# Defining Data

rollno = [1,2,3,4,5,6,7,8,9,10]
marks = [10,20,30,40,50,60,70,80,90,100]


# In[30]:


plt.scatter(rollno, marks)
plt.show()


# In[31]:


# changing color of plotted points

plt.scatter(rollno, marks, color='green')
plt.show()


# In[6]:


# changing marker of the plotted points


# In[12]:


plt.scatter(rollno, marks, color='blue', marker='*')
plt.show()   # displays the plotted graph


# In[15]:


# defining plt.figure() function

plt.figure(figsize=(12,8))
plt.scatter(rollno, marks, color='white', marker='*')
plt.show()


# In[17]:


# more functions using 'plot' function

plt.figure(figsize=(12,8))
plt.plot(rollno, marks, 'gv', markersize=10)    # gv = green color with v shaped figure
plt.show()


# In[18]:


# multiple plots on same figure


# In[19]:


temperature_pune = [25,34,21,45,28,6,43,18,7,2]       # giving data
humidity_pune = [28, 25, 29, 20, 26, 50, 19, 29, 52, 55]

temperature_banglore = [34, 35, 36, 37, 28, 27, 26, 25, 31, 20]
humidity_banglore = [40, 38, 36, 35, 42, 44, 41, 40, 34, 45]


# In[26]:


plt.figure(figsize=(8,8))
plt.plot(temperature_pune, humidity_pune, 'ro', markersize=15)
plt.show()


# In[33]:


plt.figure(figsize=(8,8))
plt.xticks(np.arange(0,60,5))
plt.yticks(np.arange(10,60,5))

plt.plot(temperature_pune, humidity_pune, 'ro', markersize=15)

plt.xlabel('Temperature')
plt.ylabel('Humidity')
plt.show()


# In[35]:


plt.figure(figsize=(8,8))
plt.xticks(np.arange(0,60,5))
plt.yticks(np.arange(10,60,5))

plt.plot(temperature_pune, humidity_pune, 'ro', markersize=15)
plt.plot(temperature_banglore, humidity_banglore, 'bo', markersize=15)

plt.xlabel('Temperature')
plt.ylabel('Humidity')
plt.show()


# # Taking Iris Dataset

# In[4]:


df = pd.read_csv('C:/Users/bhatt/Downloads/IRIS.csv')
df.head()


# In[40]:


plt.scatter(df['sepal_length'], df['petal_length'])
plt.show()


# In[41]:


plt.plot(df['sepal_length'], df['petal_width'], 'go')
plt.show()


# # Introducing Alpha(Transparency)0 - transparent 1 - opaque

# In[5]:


plt.figure(figsize=(8,8))
plt.xticks(np.arange(1,10,0.5))
plt.yticks(np.arange(1,10,0.5))

plt.plot(df['sepal_length'], df['petal_length'], 'ro', alpha=0.5, markersize=8)

plt.xlabel('sepal length')
plt.ylabel('petal length')
plt.show()


# # Line Plot

# In[43]:


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# In[44]:


plt.style.use('dark_background')


# In[45]:


rollno = [1,2,3,4,5,6,7,8,9,10]
marks = [10,20,30,40,50,60,70,80,90,100]


# # Different line styles

# In[46]:


plt.plot(rollno, marks, 'r-')
plt.show()


# In[47]:


plt.plot(rollno, marks, linestyle='-')


# In[50]:


plt.plot(rollno, marks, linestyle='-', color='#728569')


# In[49]:


plt.plot(rollno, marks, linestyle=':', color='orange')

'solid' (default) '-'
'dotted' ':'
'dashed' '-'
'dashdot' '-.'
'none'  " 'or' "
# In[51]:


plt.plot(rollno, marks, linestyle=':', linewidth=15)


# # Multiple plots on same figure

# In[55]:


study_hours=[2,3,4,4,5,6,7,7,8,9,9,10,11,11,12]
marks = [6,10,15,20,34,44,55,60,55,67,70,80,90,99,100]


# In[56]:


plt.figure(figsize=(8,8))
plt.xticks(np.arange(0,15,1))
plt.yticks(np.arange(0,100,5))

plt.plot(study_hours, marks, 'r-')

plt.xlabel('Study Hours')
plt.ylabel('Marks')
plt.show()


# In[57]:


plt.figure(figsize=(8,8))
plt.xticks(np.arange(0,15,1))
plt.yticks(np.arange(0,100,5))

plt.plot(study_hours, marks, 'r-')
plt.plot(study_hours, marks, 'bo')

plt.xlabel('Study Hours')
plt.ylabel('Marks')
plt.show()


# # Bar Plot

# In[36]:


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# In[37]:


plt.style.use('dark_background')


# In[38]:


subjects = ['Maths', 'English', 'Science', 'Social Studies', 'Computer']
marks = [89, 90, 45, 78, 99]


# In[39]:


plt.bar(subjects, marks)
plt.show()


# In[40]:


colors = ['red', 'blue', 'green', 'orange', 'purple']
plt.bar(subjects, marks, color = colors)
plt.show()


# In[41]:


plt.bar(subjects, marks, color = colors, width = 0.6, edgecolor = 'white', 
        linewidth = 4)
plt.show()


# In[42]:


plt.bar(subjects, marks, color = colors, width = 0.6, edgecolor = 'white', 
        linewidth = 3, linestyle = '-.')
plt.show()


# In[43]:


plt.barh(subjects, marks, color=colors, edgecolor='white', 
         linewidth=3, linestyle='-.')
plt.show()


# # Plotting Two Bars on the same Graph

# In[44]:


subjects = ['Maths', 'English', 'Science', 'Social', 'Computer']
marks1 = [89, 90, 45, 78, 99]
marks2 = [78, 56, 34, 90, 12]


# In[45]:


plt.figure(figsize=(8,8))

plt.bar(subjects, marks1)
plt.bar(subjects, marks2)

plt.xlabel('Subjects')
plt.ylabel('Marks')

plt.show()


# In[46]:


subjects_len = np.arange(len(subjects))
width=0.4


# In[47]:


plt.figure(figsize=(8,8))

plt.bar(subjects_len, marks1, width=width)
plt.bar(subjects_len + width, marks2, width=width)

plt.xlabel('Subjects')
plt.ylabel('Marks')
plt.show()


# In[48]:


plt.figure(figsize=(8,8))

plt.bar(subjects_len, marks1, width=width, color=colors)
plt.bar(subjects_len + width, marks2, width=width, color=colors, alpha=0.5)

plt.xlabel('Subjects')
plt.ylabel('Marks')

plt.show()


# # Plotting a bar from Supermarket Dataset

# In[49]:


df=pd.read_csv('C:/Users/bhatt/Downloads/SUPERMARKET (1).csv')
df.head()


# In[50]:


payment_df=pd.DataFrame(df['Payment'].value_counts())
payment_df


# In[51]:


colors=['red', 'blue', 'green']

plt.bar(payment_df.index, payment_df['Payment'], color=colors)
plt.show()


# # Histogram

# In[52]:


# Hist PLot


# In[165]:


import matplotlib.pyplot as ply
import numpy as np
import pandas as pd


# In[166]:


plt.style.use('dark_background')


# In[69]:


marks_50_students = np.random.randint(0, 100, (50))
marks_50_students


# In[70]:


plt.hist(marks_50_students)
plt.show()


# In[73]:


bins = np.arange(0,100,5)

plt.figure(figsize=(6,6))

plt.hist(marks_50_students, bins=bins, color = 'orange')

plt.xticks(np.arange(0,100,5))
plt.show()


# In[76]:


bins = np.arange(0,100,5)

plt.figure(figsize=(6,6))

plt.hist(marks_50_students, bins=bins, color = 'orange', orientation='horizontal')

# Changing to yticks.
plt.yticks(np.arange(0,100,5))
plt.show()


# In[77]:


bins = np.arange(0,100,5)

plt.figure(figsize=(6,6))

plt.hist(marks_50_students, bins=bins, color = 'orange', rwidth=0.6)

plt.xticks(np.arange(0,100,5))
plt.show()


# In[81]:


# Hist Types : 'bar', 'barstacked', 'step', 'stepfilled'

bins = np.arange(0,100,5)

plt.figure(figsize=(6,6))

plt.hist(marks_50_students, bins=bins, color='blue', histtype='step')

plt.xticks(np.arange(0,100,5))
plt.show()


# # Two Plots in Same Graph

# In[82]:


marks_50_students1 = np.random.randint(0, 100, (50))
marks_50_students2 = np.random.randint(0, 100, (50))


# In[86]:


bins= np.arange(0,100,5)

plt.figure(figsize=(6,6))

plt.hist([marks_50_students1, marks_50_students2], bins=bins, color=['orange', 'blue'])

plt.xticks(np.arange(0,100,5))

plt.xlabel('Marks')
plt.ylabel('Frequency')
plt.title('Marks of students of 2 classes')
plt.show()


# # Pie Plot

# In[174]:


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# In[175]:


plt.style.use('dark_background')


# In[94]:


classes = ['Physics', 'Chemistry', 'Maths', 'Science', 'SST']
marks = [89,45,78,23,90]


# In[95]:


plt.pie(marks, labels = classes)
plt.show()


# In[98]:


colors = ['red', 'blue', 'green', '#9803fc', '#03c2fc']

plt.pie(marks, labels=classes, colors=colors)
plt.show()


# In[101]:


plt.pie(marks, labels=classes, colors=colors, autopct='%0.2f%%')
plt.show()


# In[105]:


explode_values=[0.1,0.2,0,0,0]

plt.pie(marks, labels=classes, colors=colors, autopct='%0.1f%%', explode=explode_values)
plt.show()


# In[104]:


plt.pie(marks, labels=classes, colors=colors, autopct='%0.1f%%',
        explode=explode_values, shadow=True)
plt.show()


# In[108]:


plt.pie(marks, labels=classes, colors=colors, autopct='%0.1f%%',
        explode=explode_values, radius=1.6)
plt.show()


# In[109]:


textprops={'fontsize':14, 'color':'k'}

plt.pie(marks, labels=classes, colors=colors, autopct='%0.1f%%',
        explode=explode_values, radius=1.6, textprops=textprops)


# In[110]:


textprops={'fontsize':14, 'color':'k'}
wedgeprops={'linewidth':3, 'linestyle':'--', 'edgecolor':'white'}

plt.pie(marks, labels=classes, colors=colors, autopct='%0.1f%%',
        explode=explode_values, radius=1.6, textprops=textprops,
        wedgeprops=wedgeprops)
plt.show()


# In[117]:


plt.figure(figsize=(6,6))

plt.pie(marks, labels=classes, colors=colors, autopct='%0.1f%%', 
        explode=explode_values, textprops = textprops)
plt.title('Subjects and Average Scores')
plt.legend()
plt.show()


# In[119]:


df=pd.read_csv("C:/Users/bhatt/Downloads/SUPERMARKET (1).csv")
df.head()


# In[130]:


payment_df = pd.DataFrame(df['Payment'].value_counts())
payment_df


# In[131]:


plt.pie(payment_df['Payment'], labels=payment_df.index, 
        colors=['red', 'blue', 'green'], autopct='%0.2f%%')
plt.show()


# # Subplot In Matplotlib

# In[132]:


import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


# In[133]:


plt.style.use('dark_background')


# In[136]:


study_hours = [2, 3, 4, 4, 5, 6, 7, 7, 8, 9, 9, 10, 11, 11, 12]
marks = [6, 10, 15, 20, 34, 44, 55, 60, 55, 67, 70, 80, 90, 99, 100]


# In[137]:


plt.scatter(study_hours, marks)
plt.show()


# In[140]:


plt.plot(study_hours, marks, 'r-.')
plt.show()


# In[141]:


plt.hist(marks)


# In[142]:


plt.figure(figsize=(6,6))
plt.plot(study_hours, marks, 'r--')
plt.plot(study_hours, marks, 'bo')


# In[154]:


plt.figure(figsize=(8,8))

plt.subplot(2,2, 1)
plt.scatter(study_hours, marks)

plt.subplot(2,2, 2)
plt.plot(study_hours, marks, 'r--')

plt.subplot(2,2, 3)
plt.hist(marks)

plt.subplot(2,2, 4)
plt.plot(study_hours, marks, 'r--')
plt.subplot(study_hours, marks, 'bo')

# SAVING THE PLOT IN THE DESIRED LOCATION
# plt.savefig('/Desktop/subplot.png', quality = 100, facecolor='k')

plt.show()


# # Reading image using matplotlib

# In[1]:


import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np


# In[2]:


img = mpimg.imread("C:/Users/bhatt/Downloads/Amor Bhatta_Photo.jpg")

plt.imshow(img)
plt.show()


# # Reading images using OpenCV

# In[13]:


import cv2


# In[14]:


imgcv2=cv2.imread('C:/Users/bhatt/Downloads/LinkedIn-Logo.png')


# In[15]:


plt.imshow(imgcv2)
plt.show()


# In[16]:


imgcv2=cv2.cvtColor(imgcv2, cv2.COLOR_BGR2RGB)


# In[17]:


plt.imshow(imgcv2)
plt.show()


# # Changing aspect

# In[18]:


plt.imshow(img, aspect=0.5)
plt.show()


# In[22]:


img1 = mpimg.imread("C:/Users/bhatt/Downloads/Ayub.jpg")
img2 = mpimg.imread("C:/Users/bhatt/Downloads/Amor Bhatta_Photo.jpg")
img3 = mpimg.imread("C:/Users/bhatt/Downloads/supra.jpg")
img4 = mpimg.imread("C:/Users/bhatt/Downloads/Me.jpg")


# In[23]:


plt.figure()
plt.subplot(2,2,1)
plt.imshow(img1)

plt.subplot(2,2,2)
plt.imshow(img2)

plt.subplot(2,2,3)
plt.imshow(img3)

plt.subplot(2,2,4)
plt.imshow(img4)

plt.show()


# In[24]:


plt.imshow(img, aspect=1.5)
plt.colorbar()
plt.show()


# In[27]:


plt.imshow(img, cmap='PiYG_r')
plt.colorbar()
plt.show()


# In[ ]:




