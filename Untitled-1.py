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
#--------------------------------------迭代器与生成器-------------------------
#迭代器对象从集合的第一个元素开始访问，直到所有的元素被访问完结束。迭代器只能往前不会后退
list=[1,2,3,4]
it=iter(list)#创建迭代器对象
print(next(it))#输出迭代器的下一个元素
print(next(it));
import sys
# while True:
#    try:
#       print(next(it))
#    except StopIteration:
#       sys.exit();
#创建一个迭代器,把一个类作为一个迭代器使用需要在类中实现两个方法 __iter__() 与 __next__() 。
class MyNumbers:
  def __iter__(self):
    self.a = 1
    return self
 
  def __next__(self):
    if self.a<=20:
      x = self.a
      self.a += 1
      return x
    else:
      #StopIteration 异常用于标识迭代的完成，防止出现无限循环的情况，在 __next__() 方法中我们可以设置
      #在完成指定循环次数后触发 StopIteration 异常来结束迭代。
      raise StopIteration

myclass=MyNumbers()
myiter=iter(myclass)

print(next(myiter))
print(next(myiter))
print(next(myiter))
print(next(myiter))
print(next(myiter))
for x in myiter:
  print(x)
#------------生成器--------------
#在 Python 中，使用了 yield 的函数被称为生成器（generator）。
#yield 是一个关键字，用于定义生成器函数，生成器函数是一种特殊的函数，可以在迭代过程中逐步产生值，而不是一次性返回所有结果。
def countdown(n):
  while n>0:
    yield n
    n-=1
#创建生成器对象
generator=countdown(5)
#通过迭代生成器获取值
print(next(generator)) #输出：5
print(next(generator)) #输出：4
print(next(generator)) #输出：3
for value in generator:
  print(value)
 # 以上实例中，countdown 函数是一个生成器函数。它使用 yield 语句逐步产生从 n 到 1 的倒数数字。在每次调用 yield 语句时，
 # 函数会返回当前的倒数值，并在下一次调用时从上次暂停的地方继续执行。
a=0
b=1
#先计算b=b，再计算a=a+b
a,b=b,a+b 
print(a,end=" ")
print(b)
 #-------------------------------函数-------------------------
 #在 python 中，类型属于对象，对象有不同类型的区分，变量是没有类型的，仅仅是一个对象的引用（一个指针）
 #在 python 中，strings, tuples, 和 numbers 是不可更改的对象，而 list,dict 等则是可以修改的对象。
 #python 中一切都是对象，严格意义我们不能说值传递还是引用传递，我们应该说传不可变对象和传可变对象。
 #参数：1.必需参数 2. 关键字参数  3.默认参数   4.不定长参数  5.强制位置参数(3.8新增）
 #必需参数：必需参数须以正确的顺序传入函数。调用时的数量必须和声明时的一样。
def printme( str ):
   "打印任何传入的字符串"
   print (str)
   return
# 调用 printme 函数，不加参数会报错
printme("xxxxxx")
#关键字参数：使用关键字参数允许函数调用时参数的顺序与声明时不一致，因为 Python 解释器能够用参数名匹配参数值。
def printinfo( name, age ):
   "打印任何传入的字符串"
   print ("名字: ", name)
   print ("年龄: ", age)
   return
printinfo( age=50, name="runoob" )
#默认参数：如果没有传递参数，则会使用默认参数
def printinfo( name, age = 35 ):
   "打印任何传入的字符串"
   print ("名字: ", name)
   print ("年龄: ", age)
   return
printinfo( age=50, name="runoob" )
print ("------------------------")
printinfo( name="runoob" )
#不定长参数：加了星号 * 的参数会以元组(tuple)的形式导入，存放所有未命名的变量参数。如果在函数调用时没有指定参数，
# 它就是一个空元组。我们也可以不向函数传递未命名的变量,输出为空，加了**的参数会以字典的形式导入
def printinfo( arg1, *vartuple ):
   "打印任何传入的参数"
   print ("输出: ")
   print (arg1)
   print (vartuple)
printinfo( 70, 60, 50 )
#如果单独出现星号 *，则星号 * 后的参数必须用关键字传入
def f(a,b,*,c):
    return a+b+c
#f(1,2,3)   # 报错
f(1,2,c=3)
#强制位置参数 语法"/" 用来指明函数形参必须使用指定位置参数，不能使用关键字参数的形式。/之前必须指定位置参数
def f(a, b, /, c, d, *, e, f):
    print(a, b, c, d, e, f)
f(10, 20, c=30, d=40, e=50, f=60)
#匿名函数：使用lambda表达式来创建匿名函数
#一个参数
x = lambda a : a + 10
print(x(5))
#两个参数
sum = lambda arg1, arg2: arg1 + arg2
# 调用sum函数
print ("相加后的值为 : ", sum( 10, 20 ))
print ("相加后的值为 : ", sum( 20, 20 ))
#------------------------------------数据结构---------------------------------------------------
#1.列表 2.字符串 3.元组 4.集合 5.字典
#Python中列表是可变的，这是它区别于字符串和元组的最重要的特点，一句话概括即：列表可以修改，而字符串和元组不能
#列表既可以当堆栈使用，也可以当队列使用，只是队列效率不高 在列表里插入或者从头部弹出速度不快（因为所有其他的元素都得一个一个地移动）。
#每个列表推导式都在 for 之后跟一个表达式，然后有零到多个 for 或 if 子句。返回结果是一个根据表达从其后的 for 和 if 上下文环境中生成
# 出来的列表。如果希望表达式推导出一个元组，就必须使用括号。[]就代表生成一个元组
vec = [2, 4, 6]
[3*x for x in vec]
#一个元组中三个元素
#[6, 12, 18]
[[x, x**2] for x in vec]
#一个大元组包含三个小元组，小元组包含两个元素，
#[[2, 4], [4, 16], [6, 36]]
#del语句 使用 del 语句可以从一个列表中根据索引来删除一个元素，而不是值来删除元素，也可以用来删除实体变量
#------------
#集合：集合是一个无序不重复元素的集。基本功能包括关系测试和消除重复元素。可以用大括号({})创建集合。
# 注意：如果要创建一个空集合，你必须用 set() 而不是 {} ；后者创建一个空的字典
#------------
#字典：字典以关键字为索引，关键字可以是任意不可变类型，通常用字符串或数值。一对大括号创建一个空的字典：{}。
#------------
#----------------------------------模块------------------------------------------------------

