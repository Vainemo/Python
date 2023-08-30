#!/usr/bin/python3
 
count = 0
while count < 5:
   print (count, " 小于 5")
   count = count + 1
else:
   print (count, " 大于或等于 5");
#列表推导式
name=['Bob','Tom','Jerry','Wendy','Smith']
new_names=[name.upper()for name in name if len(name)>3]
print(new_names)
#字典推导式
listdemo = ['Google','Runoob', 'Taobao']
newdict={key:len(key) for key in listdemo}
print(newdict)
#集合推导式
setnew={i**2 for i in (1,2,3)}
print(setnew)
#元组推导式
a = (x for x in range(1,10))
a
tuple(a)       # 使用 tuple() 函数，可以直接将生成器对象转换成元组
print(a)