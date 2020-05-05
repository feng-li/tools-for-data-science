#!/usr/bin/env python
# coding: utf-8

# ## 关于Python
# 
# ### Python简介
# 
# Python是一种被广泛使用的高级编程语言，它的设计理念强调代码可读性，同时其语言也具有简洁性，允许程序员用更少的代码来表达概念。Python支持多种编程范式，包括面向对象的、命令式、函数式编程或面向过程的编程。它具有庞大而全面的标准库，可以轻易完成各种高级任务。

# ### 下载与安装 
# 
# 目前Python有两个版本，一个是2.x版，一个是3.x版，且两个版本不兼容。由于3.x版越来越普及，所以本教程将以最新版本Python3.7为基础。
# 
# 在Linux上，通常默认安装Python；
# 
# 在Mac系统上，如果系统自带不是3.7版本，可从Python官网下载安装；（https://www.python.org/）
# 
# 在Windows系统上，建议安装Anaconda，它是一个开源的Python发行版本。（https://www.anaconda.com/）

# ### 运行Python
# 
# Python安装成功后，打开命令提示符窗口，输入python后，看到提示符变为“>>>”就表示我们已经在Python交互式环境中了，可以输入任何Python代码，回车后会立刻得到执行结果。
# 
# 输入exit()并回车，可以退出Python交互式环境（直接关掉命令行窗口也可以）。

# ## 基本运算

# In[3]:


5 + 2


# In[4]:


5 - 2


# In[11]:


5 * 2


# In[12]:


5 ** 2  #幂


# 注意：在Python中符号(^)的用法不再是求幂，而是“异或”,用于逻辑运算。

# In[13]:


5 ^ 2


# In[14]:


5 / 2  #注意：两个整型数相除的结果是实型


# In[19]:


5 // 2  #地板除，即只取结果的整数部分


# In[6]:


5 % 2  #取余


# In[7]:


_ + 3  #在Python中，"_"可用于调用上次的结果


# ## 数据类型

# ### 字符串

# In[8]:


a = "Li Feng"
a


# In[27]:


type(a)  #type()用于求数据类型


# In[34]:


len(a)   #len()用于求字符串包含多少个字符


# In[9]:


a = "Li Feng"
a[0]  #索引从0开始


# In[35]:


a[-2]  #负号表示倒数，即从右往左数


# In[37]:


a[3:5]  #[3:5]处理为[3,5)


# In[36]:


a[3:100]  #超出索引的部分忽略


# In[33]:


a[3:]


# 字符串可以用加号连接，也可以与数字相乘，得到若干个一样的字符串

# In[21]:


b = "Li"+" "*3+"Feng"
b


# In[20]:


'6' * 3


# In[21]:


print("Li Feng")


# In[1]:


print("Hello \n World!")  #'\n'为特殊字符，表示换行


# In[38]:


print(r"Hello \n World!")  #加入r，不处理为特殊字符


# ### 列表

# In[61]:


a = [1,2,3]


# In[62]:


type(a)


# In[63]:


a[0]  #索引从0开始


# In[64]:


a.append(4)  #往list中追加元素到末尾
a


# In[65]:


a.insert(2,'a')  #把元素插入到指定的位置
a


# In[66]:


a.remove('a')  #移除列表中第一个指定元素
a


# In[67]:


b = [4,5,6]
a.extend(b)  #将两个列表合并
a


# In[68]:


a.remove(4)  
a


# In[69]:


del a[5]  #移除指定位置上的元素
a


# In[70]:


a.pop()  #移除list中的最后一个元素，并且返回该元素的值。


# In[71]:


a


# In[72]:


a.pop(2)  #移除指定位置元素，并返回该元素的值


# In[11]:


a = [1,3,2,3]
a


# In[12]:


a.sort()  #按从小到大顺序排列
a


# In[13]:


a.reverse()  #将列表顺序颠倒
a


# In[14]:


a.count(3)  #计算列表中指定元素的个数


# In[15]:


a.index(3)  #求列表中第一个指定元素的索引


# 列表的值传递与址传递：

# In[16]:


c1 = a
c2 = a[:]
c3 = a.copy()
c1,c2,c3


# In[17]:


a.append(4)
a


# In[18]:


[c1,c2,c3]  #c1与a同步变化，说明c1=a为地址传递，而c2，c3为值传递


# 列表的嵌套使用：

# In[45]:


matrix = [[1, 2, 3, 4],[5, 6, 7, 8, 9],[ 10, 11, 12]]
type(matrix)


# In[46]:


matrix[1][2]


# range经常无法使用某些方法，可以转成list进行操作：

# In[21]:


list(range(1,6,2))


# 列表生成式:把要生成的元素放到前面，后面跟for

# In[76]:


[x * x for x in range(1, 11)]


# In[77]:


[m + n for m in 'ABC' for n in 'XYZ']


# ### 集合

# In[17]:


a = {1,2,2,'a','a','bc'}  #集合中元素不重复
a


# In[3]:


type(a)


# In[73]:


'a' in a  #用in判断是否在a中，返回true 或 false


# In[5]:


'b' in a


# In[6]:


b = {1,3,'b','c'}
b


# In[12]:


a | b #求集合的并


# In[13]:


a & b  #求集合的交


# In[14]:


a - b  #求集合的差，a-b表示在a中，不在b中的元素的集合


# In[15]:


a ^ b   #求两集合的异或，a^b=(a | b)-(a & b)


# In[16]:


a = set('122abb')
a


# ### 元组

# In[18]:


a = 1,'a','b'   #元组由逗号分隔的多个值组成
a


# In[61]:


type(a)


# In[23]:


b = [1,'c']
c = a,b  #元组中可以嵌套不同类型的数据
c


# In[21]:


c[0]


# In[22]:


c[1][1]


# 元组是不可变的，但是它们可以包含可变对象。

# In[27]:


c[0] = 1


# In[28]:


c[1][1]=2
c


# ### 字典

# In[35]:


tel = {'Mike':3759, 'Mary':1462, 'Ning':6839}
print(tel)
type(tel)


# In[41]:


tel = dict(Mike = 3759, Mary = 1462, Ning = 6839)
tel


# In[60]:


tel = dict([('Mike',3759),('Mary',1462),('Ning',6839)]) #将一个由关键字与值构成的元组对序列变成字典
tel


# In[67]:


print(tel.keys())
print(tel.values())  #分别访问关键字与值


# In[70]:


list(tel.keys())


# In[68]:


sorted(tel.keys())  #排序


# In[47]:


tel['Mike']


# In[48]:


'Mike' in tel


# In[50]:


tel['Ada'] = 8080  #添加元素
tel


# In[56]:


tel['Ada'] = 8090  #修改值
tel


# In[57]:


del tel['Mary']  #删除指定元素
tel


# ##  基本语句

# ### 条件语句

# In[71]:


if True:
    print('True')  #基本语法


# In[72]:


n = 3  #判断奇偶性
if n % 2 == 0:
    print(n,'是偶数',sep = '')
elif n % 2 == 1:
    print(n,'是奇数',sep = '')
else:
    print(n,'既不是奇数也不是偶数',sep = '')


# In[74]:


#判断一个100以内的数是否为完全平方数
a=[x**2 for x in range(1,10)]
n=23
if n in a :
    print(repr(n)+' is a perfect square') #n是一个int，不可以直接用加号连上字符串，可通过repr()函数将其变为字符串
else:
    print(n,' is not a perfect square')


# ### for循环

# In[73]:


for i in range(3):
    print(i)


# continue的用法：

# In[74]:


a = {3,2,5,7,9,10,8}
for x in a:
    if x % 2 == 0:
        continue
    print(x)


# break的用法：

# In[75]:


for i in range(5):                  
    if 2 ** i < 10:
        print(i,2 ** i)
    else:
        break


# 求和：1+2+...+100

# In[76]:


a=range(1,101)  
sum=0
for s in a:
    sum=sum+s
print(sum)


# 求: 5！

# In[77]:


a=range(1,6)
factorial=1
for s in a :
    factorial=factorial*s
print(factorial)


# 求某数所有的因子：

# In[78]:


a=input('Select a number :')
divisors=[]
m=[value for value in range (1,int(a)+1)]
for s in m:
    if int(a)%s==0:
        divisors.append(s)
print(divisors)#find the set of divisors of a specific a given by users


# In[78]:


##进一步的我们可以判断一个数是否为素数
a=input('Select a number :')
divisors=[]
m=[value for value in range (1,int(int(a)**(1/2))+1)]
for s in m:
    if int(a)%s==0:
        divisors.append(s)
divisors.remove(1)
flag='true'
for divisor in divisors:
    if int(a)%divisor==0:
        flag='false'
        break
if flag=='true':
    print(a,' is a prime')
else:
    print(a,' is not a prime')
    
    


# ### while循环

# In[79]:


a = 0
while 2 ** a < 10:
    print(a,2 ** a)
    a = a + 1


# 求斐波那契数列的前n项：

# In[1]:


a=[1,1]
k=3
x=input('请输入项数(≥3)：')
while k<=int(x):
    b=a[-1]+a[-2]
    a.append(b)
    k=k+1
print(a)


# 求一个完全平方数的平方根:

# In[14]:


xx=input('Select an integer:')
x=int(xx)                    #注意xx是一个str，要进行运算必须转成int
ans=0
if  x>0:
    while ans*ans<x:
        ans=ans+1
    if ans**2==x:
        print('Its square root is '+ repr(ans))
    else:
        print('Its not a perfect square ')#来自用户的输入可能并不是完全平方数，要考虑并返回一个相应的提示
else:
    print('It is not a positive integer')


# 用while 配合 k的计数器，进行数的分组操作

# In[79]:


x=[value for value in range(1,50)]
a=['3k']
b=['3k+1']
c=['3k+2']
t=len(x)
k=1
while k<=t: #此处需要变量，t不能换为len(x)
    if x[0]%3==0:
         a.insert(0,x[0])
         x.remove(x[0])
    elif  x[0]%3==1:
         b.insert(0,x[0])
         x.remove(x[0])
    else:
         c.insert(0,x[0])
         x.remove(x[0])
    k=k+1

else:
    print(a)
    print(b)
    print(c)


# ## 导入模块及函数 

# math模块提供了许多对浮点数的数学运算函数，dir(math) 命令可以查看 math 查看包中的内容

# In[82]:


import math
math.exp(0)


# In[83]:


import math as mt
mt.exp(0)


# In[84]:


from math import exp 
exp(0) 


# In[85]:


from math import exp as myexp 
myexp(0)


# numpy（Numerical Python）提供了python对多维数组对象的支持

# In[86]:


import numpy as np
A = np.array([[1,2],[3,4]])
A


# In[87]:


A.T  #求矩阵转置


# Scipy（Scientific Python）:可以利用numpy做更高级的数学，信号处理，优化，统计等

# In[88]:


from scipy import linalg
B = linalg.inv(A) # 求矩阵的逆
B


# In[89]:


A.dot(B)   #矩阵乘法


# matplotlib：一个 Python 的 2D绘图库

# In[126]:


import matplotlib.pyplot as plt

x = [1,2,3,4,5,6]
y = [3,4,6,2,4,8]

plt.plot(x, y)


# ## 函数 

# ### 自定义函数 

# In[91]:


def parity(n):
    """To judge whether an integer is odd or even."""      # the function help
    if n % 2 == 0:
        print(n,'是偶数',sep = '')
    elif n % 2 == 1:
        print(n,'是奇数',sep = '')
    else:
        print(n,'既不是奇数也不是偶数',sep = '')


# In[92]:


help(parity)


# In[93]:


parity(3)


# In[94]:


parity(3.1)


# 匿名函数：关键字lambda表示匿名函数，冒号前面的x表示函数参数，后面只能有一个表达式，不用写return，返回值就是该表达式的结果。

# In[95]:


f = lambda x: x ** 2
f(2)


# In[84]:


def make_incrementor(n):
    return lambda x: x + n  #返回一个函数


# In[85]:


f = make_incrementor(42)


# In[86]:


f(0),f(1)


# 汉诺塔问题：定义一个函数，接收参数n，表示3个柱子A、B、C中第1个柱子A的盘子数量，然后打印出把所有盘子从A借助B移动到C的方法

# In[3]:


def move(n, a, b, c):
    if n == 1:
            print(a, '-->', c)
    else:
        move(n-1, a, c, b)
        move(1, a, b, c)
        move(n-1, b, a, c)


# In[4]:


move(3, 'A', 'B', 'C')


# 某些函数定义时设置了多个参数，使用默认参数可以简化该函数的调用：

# In[92]:


def power(x, n=2):  #幂函数
    s = 1
    while n > 0:
        s = s * x
        n = n - 1
    return s


# In[90]:


power(5) #只输入一个数，默认求其平方


# In[91]:


power(5,3)


# functools.partial可以创建一个新的函数，这个新函数可以固定住原函数的部分参数，从而在调用时更简单

# In[93]:


import functools
int2 = functools.partial(int, base=2)
int2('1000000')     #相当于int('1000000',base = 2)，即默认二进制转换为十进制


# ### 生成器(generator)

# 如果一个函数定义中包含yield关键字，那么这个函数就不再是一个普通函数，而是一个generator,可通过for循环来迭代它

# In[7]:


def triangles(n):    #杨辉三角
    L = [1]
    for x in range(n):
        yield L
        L = [1] + [L[i] + L[i+1] for i in range(len(L)-1)] + [1]


# In[8]:


for x in triangles(10):     
    print(x)


# ### 高阶函数

# 变量可以指向函数,函数名也是变量,一个函数就可以接收另一个函数作为参数，这种函数就称之为高阶函数。

# In[8]:


def add(x, y, f):
    return f(x) + f(y)


# In[9]:


add(-5, 6, abs)


# map(函数,可迭代序列)作为高阶函数，将传入的函数依次作用到序列的每个元素，并把结果作为新的迭代器返回。

# In[10]:


list(map(str, [1, 2, 3, 4, 5, 6, 7, 8, 9]))


# In[13]:


def normalize(name):  #将名字中的字母大小写规范化
    name = name.lower()
    name = name.capitalize()
    return name


# In[12]:


L = ['adam', 'LISA', 'barT']
list(map(normalize, L))


# reduce作为高阶函数，其效果是：reduce(f, [x1, x2, x3, x4]) = f(f(f(x1, x2), x3), x4)  (f必须接收两个参数)

# In[16]:


from functools import reduce
def prod(L):  #求list中所有数的乘积
    return reduce(lambda x, y: x * y, L )


# In[17]:


prod([3, 5, 7, 9])


# filter(函数，序列)：把传入的函数依次作用于每个元素，然后根据返回值是True还是False决定保留还是丢弃该元素。

# In[6]:


list(filter(lambda x: x % 2 == 1, [1, 2, 4, 5, 6, 9, 10, 15]))  #返回list中的奇数


# sorted(序列，keys)：按照keys中函数作用后的结果进行排序，并按照对应关系返回list相应的元素

# In[3]:


sorted([36, 5, -12, 9, -21], key=abs)


# In[5]:


students = [('Bob', 75), ('Adam', 92), ('Bart', 66), ('Lisa', 88)]

print(sorted(students, key=lambda x: x[0]))  #按名字
print(sorted(students, key=lambda x: x[1]))  #按成绩从低到高
print(sorted(students, key=lambda x: x[1], reverse=True))  #按成绩从高到低


# ## Python 的类

# 面向对象的程序设计思想，是把对象作为程序的基本单元：类是抽象的模板，实例是根据类创建出来的一个个具体的“对象”，每个对象都拥有相同的方法，但各自的数据可能不同。

# In[104]:


class MyClass:
    """A simple example class"""
    i = 12345
    def f(self):
        return 'hello world'


# In[95]:


MyClass()


# In[96]:


MyClass.i  #引用属性


# In[97]:


MyClass.f


# In[98]:


MyClass.i = 3  #更改属性值
MyClass.i


# In[103]:


MyClass.x = 1  #根据需要添加定义中没有的属性
MyClass.x


# 在创建实例的时候，定义一个特殊的__init__方法，把一些我们认为必须绑定的属性强制填写进去，可以起到模板的作用。

# In[20]:


class Complex:
    def __init__(self, realpart, imagpart):  #注意：特殊方法“__init__”前后分别有两个下划线
        self.r = realpart
        self.i = imagpart


# In[21]:


x = Complex(3.0, -4.5)
x.r, x.i


# ##  读取文件 

# ### 读取txt

# In[100]:


pwd


# 在上述目录下创建一个test.txt,写入“Hello world！”

# In[24]:


file_for_reading = open('test.txt', 'r')  #‘r’表示read


# In[2]:


file_for_reading.read()


# In[3]:


file_for_reading.close()


# In[25]:


file_for_writing = open('test.txt', 'w')  #‘w’表示write


# In[5]:


file_for_writing.write('I love studying! \n')


# In[6]:


file_for_writing.close()


# 查看test.txt，发现内容变成了‘I love studying!’，说明原内容被覆盖

# In[26]:


file_for_appending = open('test.txt','a') #‘a’表示append


# In[8]:


file_for_appending.write('Hello world! \n')


# In[27]:


file_for_appending.close()


# 再次查看，发现原内容后加入了一行Hello world! 

# 由于close()很容易忘记，故推荐采用with语句，在语句执行完毕后自动关闭：

# In[10]:


with open('test.txt','a') as file:
    file.write('Nice to meet you! \n')


# ### 读取csv

# 在工作目录下创建一个stocks.csv,由symbol,date,closing_price三列构成，并填充数据

# In[12]:


import csv

data = {'symbol':[], 'date':[], 'closing_price' : []}
with open('stocks.csv', 'r') as f:
    reader = csv.DictReader(f)
    for row in reader:
        data['symbol'].append(row["symbol"])
        data['date'].append(row["date"])
        data['closing_price'].append(float(row["closing_price"]))


# In[112]:


data.keys()


# In[113]:


data['closing_price']


# 也可使用pandas包中的read_csv()函数读取csv文件：

# In[13]:


import pandas  
data2 = pandas.read_csv('stocks.csv') 
print(len(data2)) 
print(type(data2))


# In[14]:


data2


# In[16]:


data2.iloc[1]


# In[17]:


data2.iloc[1]['date']


# ## 文本处理 

# In[116]:


import nltk


# In[117]:


nltk.download('punkt')


# 分段为句：

# In[118]:


para = "Python is a widely used general-purpose, high-level programming language. Its design philosophy emphasizes code readability, and its syntax allows programmers to express concepts in fewer lines of code than would be possible in languages such as C++ or Java."


# In[119]:


from nltk.tokenize import sent_tokenize 
sent_tokenize(para)


# 分段为词：

# In[120]:


from nltk.tokenize import word_tokenize
word_tokenize(para)


# 过滤掉语句中的“stopwords”：

# In[122]:


nltk.download('stopwords')


# In[28]:


from nltk.corpus import stopwords
english_stops = set(stopwords.words('english'))
print(english_stops)  #输出stopwords


# In[121]:


from nltk.tokenize import RegexpTokenizer 
tokenizer = RegexpTokenizer("[\w']+") 
words = tokenizer.tokenize("Smoking is now banned in many places of work.") 
words


# In[124]:


[word for word in words if word not in english_stops]


# 去掉词缀：

# In[125]:


from nltk.stem import PorterStemmer
stemmer = PorterStemmer()
stemmer.stem('cooking')

