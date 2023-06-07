# -*- coding: utf-8 -*-
"""
Created on Thu Feb 16 22:34:26 2023

@author: ayaha
"""

"""      GRAPHING TRAINING VALUES      """
fig2 = figure()
ax2=fig2.add_subplot()
col2 = []
for i in range(0,len(y_train)):
  if y_train[i]==0:
    col2.append('blue')
  if y_train[i]==1:
    col2.append('green')  
X_trainT = transpose(X_train)
Xg, Yg = (X_trainT)
for i in range(len(y_train)):
  ax2.scatter(Xg[i], Yg[i], c=col2[i])
ax2.set_xlabel('X Label')
ax2.set_ylabel('Y Label')
ax2.set_title('Training Set Plot')
show()
"""      GRAPHING TEST VALUES      """   
fig3 = figure()
ax3=fig3.add_subplot()
col3 = []
for i in range(0,len(y_test)):
  if y_test[i]==0:
    col3.append('blue')
  if y_test[i]==1:
    col3.append('green')  
X_testT = transpose(X_test)
Xg2, Yg2 = (X_testT)
for i in range(len(y_test)):
  ax3.scatter(Xg2[i], Yg2[i], c=col3[i])
ax3.set_xlabel('X Label')
ax3.set_ylabel('Y Label')
ax3.set_title('Test Set Plot')
show()

print(Xg2,Yg2)