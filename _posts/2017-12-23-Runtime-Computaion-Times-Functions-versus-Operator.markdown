---
layout: post
title: "Runtime Computation Times - Functions versus Operator"
date: 2017-12-23 10:37:00
categories: Write-ups
tags: [Python, functions, operators, time, computation, runtime, computation time, execution, execution time, functions vs operators]
comments: true
---

A test on integer and float exponentials can be run to get the computation times.

```python
print(timeit.timeit(stmt="pow(2, 100)"))
1.30916247735
```

```python
print(timeit.timeit(stmt="pow(2, 1023)"))
3.69032251306
```

```python
print(timeit.timeit(stmt="math.pow(2, 100)", setup='import math'))
0.322029196449
```

```python
print(timeit.timeit(stmt="math.pow(2, 1023)", setup='import math'))
0.334137509097
```

```python
print(timeit.timeit(stmt="pow(2.01, 1016)"))
0.302062482802
```

```python
print(timeit.timeit(stmt="math.pow(2.0, 1023)", setup='import math'))
0.310684528341
```

```python
print(timeit.timeit(stmt="math.pow(2.01, 1016)", setup='import math'))
0.310034306037
```

Now, see the fall while using the operator.

```python
print(timeit.timeit(stmt="2 ** 1023"))
0.0310322261626
```

```python
print(timeit.timeit(stmt="2.0 ** 1023"))
0.0324852041249
```

```python
print(timeit.timeit(stmt="2.01 ** 1016"))
0.0302783594007
```

```python
print(timeit.timeit(stmt="2.01 ** 1016.01"))
0.0301967149462
```

This apparent time difference at runtime occurs most likely due to the overhead function call procedure. Disassembling to bytecodes gives an insight.

```python
dis.dis('2.01 ** 1016')
  1           0 LOAD_CONST               2 (1.1147932725682862e+308)
              2 RETURN_VALUE
```

```python
dis.dis('pow(2.01, 1016)')
  1           0 LOAD_NAME                0 (pow)
              2 LOAD_CONST               0 (2.01)
              4 LOAD_CONST               1 (1016)
              6 CALL_FUNCTION            2
              8 RETURN_VALUE
```

```python
dis.dis('math.pow(2.01, 1016)')
  1           0 LOAD_NAME                0 (math)
              2 LOAD_ATTR                1 (pow)
              4 LOAD_CONST               0 (2.01)
              6 LOAD_CONST               1 (1016)
              8 CALL_FUNCTION            2
             10 RETURN_VALUE
```

Also, as a food for thought, notice that while the performance of the inbuilt `pow` function depends upon the numerics given in the arguments, the `math.pow` acts mostly indifferent.
