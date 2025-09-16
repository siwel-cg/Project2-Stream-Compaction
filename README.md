CUDA Stream Compaction
======================

**University of Pennsylvania, CIS 565: GPU Programming and Architecture, Project 2**

* Lewis Ghrist  
* [LinkedIn](https://www.linkedin.com/in/lewis-ghrist-4b1b3728b/), [Personal Website](https://siwel-cg.github.io/siwel.cg_websiteV1/index.html#home)  
* Tested on: Windows 11, AMD Ryzen 9 5950X 16-Core Processor, 64GB, NVIDIA GeForce RTX 3080 10GB (Personal PC)

---

## Overview
Often when thinking about parallelization, we think of use cases that are already inherintly parallel by nature in that each calculation doesn't really depend on eachother. Things like running a series calculations on every particle or each pixel of the screen. With these, we can ofcourse assign each element to a thread, and do the calculations all at once. However, what about problems which seem be purely sequenctial? In this project, I implement two such problems, scan and stream compaction, which take advantage of GPU parallelization to reduce the total number of operations performed. A naive CPU version was also implemneted for comparisson.
### Scan
The Scan problem is simply: given an array of elements and some binarry operation which can combine two elements, we want to return the "sum" of any element with all the elements before it. In this implementation we we looking at specifically summing an array of integers, however any group can work (see later note on groups). There are two different types of the scan algorithm: Exclusive Scan and Inclusive Scan. In an Exclusive Scan, index i in our result array will contain the sum of all elements with idices stricly less than i. That is, the ith element in our input will not be included in the sum stored in the ith element of our output. This means our first output element will always be 0 and the last element in our input is never added and is irelivant to the output. For an Inclusive Scan, we do take into account the ith element in our sum. This means the first element of our output is always the first elment of our input, and the last element of our output is the sum of all elements in the array. In this project I implemented Exclusive Scans, but it is simple to convert between the two types deppending on the use case.

\underline{CPU Scan}

This was implemented as a simple for loop which takes the first element

\underline{Naive Parallel Scan}

If we expand out each sum in our sequenctial sum, we see that we can break down that entire sum into pairs of sums, then one again pairs of sums those, and so on untill finally we have just one value left. The result looks like a binary tree where at each level we do half the number of adds of the previous level. Note also that at each level, we have actually already completed a sum for some of the outputs. The advantage of doing things this way is that each individual sum in a layer doesn't rely on the others, and can thus be computed in parallel. In my implementation, since we are reading and writing to the same array, I used a ping-pong array system for the output to avoid any incorrect adds. Overall, we get O(nlog_2(n)) adds and O(log_2(n)) 

![Naive Scan](img/NaiveScan_V1.png)

\underline{Work-Efficient Parallel Scan}
Although it might not be obvious at first we can actually cut out some of the adds, but 

![Up Sweep](img/UpSweep_V1.png)

![Down Sweep](img/DownSweep_V1.png)

### Stream Compaction

### A Quick Note On Groups
As mentioned earlier, with the Scan algorithm, and thus Stream Compaction aswell, we are simply given an array of elements and some operation that can be used on those elements. If we look further into the algorithm however, we see that two more conditions need to be met: our operation must be associative and there must be an identity element. This is precicely the axioms needed for a group! Although in practice we would need to do a bit more work and memory managment if our group element isn't a simple number that can be stored in an array, the algorithm will still work as expected. There is just one other thing we need to be careful of when implementing Down Sweep in the Work Efficient Scan: Commutativity. For many groups, including our intager addition we implemented, our group is abelian, which means A+B = B+A as you would expect. However this is not the case for all groups. For example, matrix multiplications is not-commutative and thus we need to be extra cearful when multiplying to ensure we maintain the ordering of or initial array. As presented in the pictures above, when doing the swap/combine step of two elements, doing Left * Right will actually break the order of our input array. However, if we do Right * Left at each step, we see the ordering gets maintained, meaning that we can still use these efficient methods for non-abelian groups. 

---
## Results

--- 

## Performance Analysis

## References
- [Nvidia Developer GPU Gems 3 ](https://developer.nvidia.com/gpugems/gpugems3/part-vi-gpu-computing/chapter-39-parallel-prefix-sum-scan-cuda)