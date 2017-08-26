---
title: "Introduction to DSP using MATLAB - Part I"
date: 2017-04-09 00:00:00
categories: DSP
tags: [DSP, MATLAB, convolution, system, frequency, FIR]
---
Last semester, I had to take the course of Digital Signal Processing as a part of my ECE undergrad coursework. Unlike many other papers, to one's awe, I didn't find any reference site or portal, that would give me an overall review of the subject from an application-oriented viewpoint before being bogged down with its theories and pracical assignments. Or that I could look upto, use as a reference site, to back up the learning process of this course throughout the semester.

The idea is to take an application-oriented practical approach towards the beautiful subject of DSP, that would help new learners taste its power, benefits before the theoritical knowledge takes over to complete it. Although, for gaining deep insights I do recommend thorough studying of text books, for which I would suggest the one authored by John G. Proakis, Manokalis and the one by Sir Alan V. Oppenheim, which you can buy on Amazon from [here][book1] and [here][book2].

Another purpose of this series of blog posts will be to function as an online reference for myself and other professionals in this field. I have tried to put together MATLAB algorithms of most of the topics covered in the standard DSP course at MIT OpenCourseWare. That being the reason, to keep the blog post from being extremely long and to keep you interested, I have split it in 3 segments along with a [reference blog post][part4], that indexes and describes shortly the set of functions I have used in this series and further readings and references.



### Generation of Discrete-Time Sequences


This one is pretty straight forward. We use the mathematical functions for each of the known waveforms and `subplot` them in four sectors. In case of the unit step function we create a row matrix of value 1 using the `ones` function.

```
%% Generation of Discrete-Time Sequences
% Unit Step Sequence
N = 21;
x = ones(1,N);
n = 0:1:N-1;
subplot(2,2,1), stem(n,x);
xlabel('n'),ylabel('x(n)');
title('Unit Step Sequence');
%Sinusoidal Sequence
x1 = cos(.2*pi*n);
subplot(2,2,2), stem(n,x1);
xlabel('n'),ylabel('x1(n)');
title('Sinusoidal Sequence')
%Exponential Sequence
x2 = .8.^(n);
subplot(2,2,3), stem(n,x2);
xlabel('n'),ylabel('x2(n)');
title('Exponential Sequence')
%Addition of two Sinusoidal Sequence
x3 = sin(.1*pi*n) + sin(.2*pi*n);
subplot(2,2,4), stem(n,x3);
xlabel('n'),ylabel('x3(n)');
title('Addition of two Sinusoidal Sequences');
```

![Alt](/images/dsp_matlab_1/blog1.jpg "Output waveforms of discrete-time sequences")




---------------------------------------------------------------------
### **Under maintenance. Please check back soon.**
---------------------------------------------------------------------





### Convolution of Two Sequences

//text

```

```

![Alt](/images/dsp_matlab_1/blog2.jpg "Output waveform of convolution of two sequences")


### Frequency Response of a First Order System

//text

```

```

![Alt](/images/dsp_matlab_1/blog3.jpg "Output frequency response waveform of a first order system")


### Frequency Response of a Given System

//text

```

```

![Alt](/images/dsp_matlab_1/blog4.jpg "Output frequency response waveform of a given system")


### Frequency Response of an FIR System

//text

```

```

![Alt](/images/dsp_matlab_1/blog5.jpg "Output frequency response waveform of an FIR system")


### Periodic and Aperiodic Sequences

//text

```

```

![Alt](/images/dsp_matlab_1/blog6.jpg "Output waveform of periodic and periodic sequences")


### Periodicity Property of Digital Frequency

//text

```

```

![Alt](/images/dsp_matlab_1/blog7.jpg "Waveforms that explain periodicity property of digital frequency")


### Demostration of the Property of Digital Frequency

//text

```

```

![Alt](/images/dsp_matlab_1/blog8.jpg "Waveforms that demonstrate the property of digital frequency")


### A Notch Filter that filters 50 Hz Noise

//text

```

```

![Alt](/images/dsp_matlab_1/blog7.jpg "Waveforms of a notch filter that filters 50 Hz noise")


[book1]:		http://www.amazon.in/Digital-Signal-Processing-Principles-Applications/dp/8131710009/ref=sr_1_fkmr0_1?ie=UTF8&qid=1493404937&sr=8-1-fkmr0&keywords=proakis+manokalis
[book2]:		http://www.amazon.in/Digital-Signal-Processing-Oppenheim-Schafer/dp/9332550336/ref=sr_1_2?ie=UTF8&qid=1493405049&sr=8-2&keywords=oppenheim
[part4]:		http://sohambhattacharyya.me/2017/Introduction-to-DSP-using-MATLAB-Part-I/