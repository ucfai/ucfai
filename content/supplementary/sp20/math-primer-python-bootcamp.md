---
title: "A Math Refresher and Scientific Computing Crash Course"
linktitle: "A Math Refresher and Scientific Computing Crash Course"

date: "2020-01-23"
lastmod: "2020-01-23"

draft: false
toc: true
type: docs

weight: 1

menu:
  supplementary_sp20:
    parent: Spring 2020
    weight: 1

authors: ["calvinyong", "brandons209", ]

urls:
  youtube: ""
  slides:  "https://docs.google.com/presentation/d/1UifKinX6Cd3DK_EetHIpKdf9rsliLwoQVfWRnzGz3UI"
  github:  "https://github.com/ucfai/supplementary/blob/master/sp20/2020-01-23-math-primer-python-bootcamp/2020-01-23-math-primer-python-bootcamp.ipynb"
  kaggle:  "https://kaggle.com/ucfaibot/supplementary-sp20-math-primer-python-bootcamp"
  colab:   "https://colab.research.google.com/github/ucfai/supplementary/blob/master/sp20/2020-01-23-math-primer-python-bootcamp/2020-01-23-math-primer-python-bootcamp.ipynb"

categories: ["sp20"]
tags: ["python", "basics", "linear-algebra", "calculus", "derivatives", "gradients", ]
description: >-
  We'll be covering the foundational mathematics for Machine Learning, spanning Multivariate Calculus to Linear Algebra, with a sprinkling of Statistics. Following this, we'll be reviewing the basics of Python concerning tools, syntax, and data structures.
---
```
# Hello World!
print("Hello, World!");
```

    Hello, World!


The purpose of this tutorial is to get you up and running with Python 3 syntax. This tutorial is aimed for students with no prior experience in Python 3. This is by no means a comprehensive guide. It serves as a concise but full-featured tutorial. See the [official documentation](https://docs.python.org/3/tutorial/) for reference.

If youre coming from a strong Java background, [this](http://python4java.necaiseweb.org/Main/TableOfContents) might be a helpful guide.


# Installation

* Mac:
MacOS comes pre-installed with Python 3.
See https://docs.python.org/3/using/mac.html for additional help

* Windows:
See https://docs.python.org/3/using/windows.html for additional help

* Our Python 3 development will be primarily in Google Colab so I will not be going over detailed installation instructions. See https://towardsdatascience.com/getting-started-with-google-colab-f2fff97f594c for a quick tutorial on Colab. 

# Primitive Datatypes

In Python, there is no need to declare variables and therefore there is no need to declare type either. You hardly need to bother about types but you may need to explicitly cast from time to time.

Here is a list of Python datatypes:
* float - used for real numbers.
* int - used for integers.
* str - used for texts. We can define strings using single quotes'value', double quotes"value", or triple quotes"""value""". The triple quoted strings can be on multiple lines, the new lines will be included in the value of the variable. They’re also used for writing function documentation.
* bool - used for truthy values. Useful to perform a filtering operation on a data.
* list - used to store a collection of values.
* dict - used to store a key-values pairs.

```
x = 3              # a whole number                   
f = 3.141          # a floating point number              
name = "Python"    # a string

print(x)
print(f)
print(name)

combination = name + " " + str(x) # explicit conversion from int to string
print(combination)

sum = f + x # implicit conversion from int to float
print(sum)
```

    3
    3.1415926
    Python
    Python 3
    6.1415926


```
# Python implements all of the usual operators for Boolean logic, but uses English words rather than symbols 
t = True
f = False
print(type(t)) # Prints "<class 'bool'>"
print(t and f) # Logical AND; prints "False"
print(t or f)  # Logical OR; prints "True"
print(not t)   # Logical NOT; prints "False"
print(t != f)  # Logical XOR; prints "True"
    
```

    <class 'bool'>
    False
    True
    False
    True


```
# String objects have a bunch of useful methods; for example:
s = "hello"
print(s.capitalize())  # Capitalize a string; prints "Hello"
print(s.upper())       # Convert a string to uppercase; prints "HELLO"
print(s.rjust(7))      # Right-justify a string, padding with spaces; prints "  hello"
print(s.center(7))     # Center a string, padding with spaces; prints " hello "
print(s.replace('l', '(ell)'))  # Replace all instances of one substring with another;
                                # prints "he(ell)(ell)o"
print('  world '.strip())  # Strip leading and trailing whitespace; prints "world"
```

    Hello
    HELLO
      hello
     hello 
    he(ell)(ell)o
    world


# Operators

Python special symbols to carry out arithmetic operaitons on variables. This is pretty standard for other languages you have encountered. 

```
x = 14
y = 4

# Add two operands
print('x + y =', x+y) # Output: x + y = 18

# Subtract right operand from the left
print('x - y =', x-y) # Output: x - y = 10

# Multiply two operands
print('x * y =', x*y) # Output: x * y = 56

# Divide left operand by the right one 
print('x / y =', x/y) # Output: x / y = 3.5

# Floor division (quotient)
print('x // y =', x//y) # Output: x // y = 3

# Remainder of the division of left operand by the right
print('x % y =', x%y) # Output: x % y = 2

# Left operand raised to the power of right (x^y)
print('x ** y =', x**y) # Output: x ** y = 38416

# Python also has standard assignment operators
x = 5

# x += 5 ----> x = x + 5
x +=5
print(x) # Output: 10

# x /= 5 ----> x = x / 5
x /= 5
print(x) # Output: 2.0
```

    x + y = 18
    x - y = 10
    x * y = 56
    x / y = 3.5
    x // y = 3
    x % y = 2
    x ** y = 38416
    10
    2.0


# Types

The data structures available in python are lists, tuples and dictionaries.

### Lists

A list is created by placing all the items (elements) inside a square bracket `[]` separated by commas.

It can have any number of items and they may be of different types (`integer`, `float`, `string` etc.)



#### List Creation

```
# empty list
my_list = []
# list of integers
my_list = [1, 2, 3]
# list with mixed data types
my_list = [1, "Hello", 3.4]
```

#### List Methods

```
# Start with an empty list
li = []

# Add stuff to the end of a list with append
li.append(1)  # li is now [1]
print(li)
li.append(2)  # li is now [1, 2]
print(li)
li.append(4)  # li is now [1, 2, 4]
print(li)
li.append(3)  # li is now [1, 2, 4, 3]
print(li)
# Remove from the end with pop
li.pop()  # => 3 and li is now [1, 2, 4]
print(li)

# Access a list like you would any array
print(li[0])  # => 1
# Assign new values to indexes that have already been initialized with =
li[0] = 42
print(li[0])  # => 42
# Look at the last element
print(li[-1])  # => 4

# Looking out of bounds is an IndexError
# li[4]  # Raises an IndexError

# You can look at ranges with slice syntax.
# (It's a closed/open range for you mathy types.)
print(li[1:3])  # => [2, 4]
# Omit the beginning
print(li[2:])  # => [4, 3]
# Omit the end
print(li[:3])  # => [1, 2, 4]
# Select every second entry
print(li[::2])  # =>[1, 4]
# Reverse a copy of the list
print(li[::-1])  # => [3, 4, 2, 1]
# Use any combination of these to make advanced slices
# li[start:end:step]

# Remove arbitrary elements from a list with "del"
del li[2]  # li is now [1, 2, 3]
print(li)


# You can add lists
other_li = [4, 5, 6]
new_li = li + other_li  # => [42, 2, 4, 5, 6]
print(new_li)
# Note: values for li and for other_li are not modified.

# Concatenate lists with "extend()"
li.extend(other_li)  # Now li is [42, 2, 4, 5, 6]
print(li)

# Remove first occurrence of a value
li.remove(2)  # li is now [42, 4, 5, 6]
print(li)
# li.remove(2)  # Raises a ValueError as 2 is not in the list

# Insert an element at a specific index
li.insert(1, 2)  # li is now [42, 2, 4, 5, 6] again
print(li)

# Get the index of the first item found
print(li.index(2))  # => 1
# li.index(7)  # Raises a ValueError as 7 is not in the list

# Check for existence in a list with "in"
print(5 in li)  # => True

# Examine the length with "len()"
print(len(li))  # => 5

# **IMPORTANT**
# When you create a new array that is referenced to an existing one, they
# are poiinting to the same memory. Therefore, changing a value in the
# new array will change the original array.
new_list = li
new_list[0] = 12
print(li)
```

    [1]
    [1, 2]
    [1, 2, 4]
    [1, 2, 4, 3]
    [1, 2, 4]
    1
    42
    4
    [2, 4]
    [4]
    [42, 2, 4]
    [42, 4]
    [4, 2, 42]
    [42, 2]
    [42, 2, 4, 5, 6]
    [42, 2, 4, 5, 6]
    [42, 4, 5, 6]
    [42, 2, 4, 5, 6]
    1
    True
    5
    [12, 2, 4, 5, 6]


#### List Comprehension

List comprehensions provide us with a simple way to create a list based on some iterable. During the creation, elements from the iterable can be conditionally included in the new list and transformed as needed.
An iterable is something you can loop over.

`new_list = [expression for member in iterable]`

The components of a list comprehension are:
* **expression** is the member itself, a call to a method, or any other valid expression that returns a value. In the example below, the expression i * i is the square of the member value.
* **member** is the object or value in the list or iterable. In the example below, the member value is i.
* **iterable** is a list, set, sequence, generator, or any other object that can return its elements one at a time. In the example below, the iterable is range(10)


```
squares = [i * i for i in range(10)]
print(squares)
```

    [0, 1, 4, 9, 16, 25, 36, 49, 64, 81]


##### Using Conditional Logic
`new_list = [expression for member in iterable (if conditional)]`

```
sentence = 'the rocket came back from mars'
vowels = [i for i in sentence if i in 'aeiou']
print(vowels)
```

    ['e', 'o', 'e', 'a', 'e', 'a', 'o', 'a']


You can also create create more complex filters using custom functions and pass this function as the conditional statement for your list comprehension. The member value is also passed as an argument to your function.

### Tuples
Tuple is similar to a list except you cannot change elements of a tuple once it is defined (immutable). Whereas in a list, items can be modified. Tuples can contain anything, including lists! One of the most important differences is that tuples can be used as keys in dictionaries and as elements of sets, while lists cannot.

```
vowels = ('a', 'e', 'i', 'o', 'i', 'o', 'e', 'i', 'u')
print(vowels)

# You can access elements of a tuple in a similar way like a list.
print(vowels[2])

# Tuples contain many methods, such as count.
print(vowels.count('i'))
```

    ('a', 'e', 'i', 'o', 'i', 'o', 'e', 'i', 'u')
    i
    3


### Sets

A set is an unordered collection of items where every element is unique (no duplicates). Sets are mutable. You can add, remove and delete elements of a set. However, you cannot replace one item of a set with another as they are unordered and indexing have no meaning.



```
# set of integers
my_set = {1, 2, 3}

my_set.add(4)
print(my_set) # Output: {1, 2, 3, 4}

my_set.add(2)
print(my_set) # Output: {1, 2, 3, 4}

my_set.update([3, 4, 5])
print(my_set) # Output: {1, 2, 3, 4, 5}

my_set.remove(4)
print(my_set) # Output: {1, 2, 3, 5}
```

    {1, 2, 3, 4}
    {1, 2, 3, 4}
    {1, 2, 3, 4, 5}
    {1, 2, 3, 5}


### Dictionaries

Dictionary is an unordered collection of items. They are mutable pairs of key/value. The key must be immutable type such as strings or numbers. There are defined using curly brackets.

We can add, update, and delete data from our dictionaries. When we want to add or update the data we can simply use this code `our_dict[key] = value`. When we want to delete a key-value pair we do this like that `del(our_dict[key])`.

```
# empty dictionary
my_dict = {}

# dictionary with integer keys
my_dict = {1: 'apple'}
# Access an key in the dictionary
print(my_dict[1])
# Add a key, value pair
my_dict[2] = 'ball'
print(my_dict)

# dictionary with mixed keys
my_dict = {'name': 'John', 1: [2, 4, 3]}

# Delete name
del my_dict['name']
print(my_dict)

# Delete entire dictionary
del my_dict
```

    apple
    {1: 'apple', 2: 'ball'}
    {1: [2, 4, 3]}


```
# Dictionary comprehensions: These are similar to list comprehensions, but allow you to easily construct dictionaries
```

```
nums = [0, 1, 2, 3, 4]
even_num_to_square = {x: x ** 2 for x in nums if x % 2 == 0}
print(even_num_to_square)  # Prints "{0: 0, 2: 4, 4: 16}"
```

    {0: 0, 2: 4, 4: 16}


# Control Flow

Indentation is Python is an essential aspect in understanding the syntax. You will see in the following control statements that indentaiton delimits the scope of the loop. 

#### if...else statement

There can be zero or more elif parts, and the else part is optional.

```
num = -1

if num > 0:
  print("Positive number")
elif num == 0:
  print("Zero")
else:
  print("Negative number")
    
```

    Negative number


### while loop

```
n = 100

# initialize sum and counter
sum = 0
i = 1

while i <= n:
    sum = sum + i
    i = i+1    # update counter

print("The sum is", sum)

# Output: The sum is 5050
```

    The sum is 5050


### for loop

for loop is used to iterate over a sequence (list, tuple, string) or other iterable objects

```
numbers = [6, 5, 3, 8, 4, 2]

summ = 0

# iterate over the list
for val in numbers:
  summ = summ + val

print("The sum is", summ) # Output: The sum is 28
```

    The sum is 28


### break

The break statement terminates the loop containing it. Control of the program flows to the statement immediately after the body of the loop.

```
for val in "string":
  if val == "r":
    break
  print(val)

print("The end")
```

    s
    t
    The end


### continue

The continue statement is used to skip the rest of the code inside a loop for the current iteration only. Loop does not terminate but continues on with the next iteration.

```
for val in "string":
    if val == "r":
        continue
    print(val)

print("The end")
```

    s
    t
    i
    n
    g
    The end


# OOP

Everything in Python is an object including integers, floats, functions, classes, and `None`. 

### Functions

In Python, you can define a function that takes variable number of arguments. You will learn to define such functions using default, keyword and arbitrary arguments

```
# User-defined arguments
def sum_two(a, b):
  return a + b

print(sum_two(2, 3))
```

    5


```
# Default arguments 
# Note, default arguments must come after non-default arguments.
def sum_two(a, b = 3):
  return a + b

print(sum_two(2))
```

    5


```
# Sometimes, we do not know in advance the number of arguments that 
# will be passed into a function. This notation allows for an arbitrary number of arguments.
def sum_two(*sums):
  summ = 0
  for i in sums:
    summ = summ + i
  return summ

print(sum_two(2, 3, 4))
```

    9


```
# Lastly, Python allows functions to be called using keyword arguments. 
# When we call functions in this way, the order (position) of the arguments can be changed.
def sum_two(a, b):
  return a + b

print(sum_two(b = 2, a = 1))
```

    3


### Lambda Function

In Python, you can define functions without a name. These functions are called lambda or anonymous function.

```
square = lambda x: x ** 2
print(square(5))
```

    25


### Classes

Python classes are defined with the `class` keyword. As soon as you define a class, a new class object is created with the same name. This class object allows us to access the different attributes as well as to instantiate new objects of that class.

```
class myClass():
  a = 10
  def func(self):
    print('Hello')

my_class = myClass()
print(my_class.a)
my_class.func()
```

    10
    Hello


### Constructors

In Python, a method with name `__init()__` is a constructor. This method is automatically called when an object is instantiated.

```
class myClass():
  def __init__(self, num = 0):  # constructor
    self.num = num

my_class = myClass(12)
print(my_class.num)
```

    12


### Inheritance

Inheritance is a feature used in OOP; it refers to defining a new class with less or no modification to an existing class. The new class is called derived class and from one which it inherits is called the base. Python supports inheritance; it also supports multiple inheritances. A class can inherit attributes and behavior methods from another class called subclass or heir class.

```
class Mammal:
  def displayMammalFeatures(self):
    print('Mammal is a warm-blooded animal.')

class Dog(Mammal):
  def displayDogFeatures(self):
    print('Dog has 4 legs.')

d = Dog()
d.displayDogFeatures()
d.displayMammalFeatures()
```

    Dog has 4 legs.
    Mammal is a warm-blooded animal.


# References

* https://www.learnpython.org/
