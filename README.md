CUDA Stream Compaction
======================

**University of Pennsylvania, CIS 565: GPU Programming and Architecture, Project 2**

* Lewis Ghrist  
* [Personal Website](https://siwel-cg.github.io/siwel.cg_websiteV1/index.html#home), [LinkedIn](https://www.linkedin.com/in/lewis-ghrist-4b1b3728b/)
* Tested on: Windows 11, AMD Ryzen 9 5950X 16-Core Processor, 64GB, NVIDIA GeForce RTX 3080 10GB

---

## Overview
Often when thinking about parallelization, we think of use cases that are already inherintly parallel by nature in that each calculation doesn't really depend on eachother. Things like running a series calculations on every particle or each pixel of the screen. With these, we can ofcourse assign each element to a thread, and do the calculations all at once. However, what about problems which seem be purely sequenctial? In this project, I implement two such problems, scan and stream compaction, which take advantage of GPU parallelization to reduce the total number of operations performed. A naive CPU version was also implemneted for comparisson.
### Scan
The Scan problem is simply: given an array of elements and some binarry operation which can combine two elements, we want to return the "sum" of any element with all the elements before it. In this implementation we we looking at specifically summing an array of integers, however any group can work (see later note on groups). There are two different types of the scan algorithm: Exclusive Scan and Inclusive Scan. In an Exclusive Scan, index i in our result array will contain the sum of all elements with idices stricly less than i. That is, the ith element in our input will not be included in the sum stored in the ith element of our output. This means our first output element will always be 0 and the last element in our input is never added and is irelivant to the output. For an Inclusive Scan, we do take into account the ith element in our sum. This means the first element of our output is always the first elment of our input, and the last element of our output is the sum of all elements in the array. In this project I implemented Exclusive Scans, but it is simple to convert between the two types deppending on the use case.

$\underline{CPU Scan}$

This was implemented as a simple for loop which adds the current element with the result of the previous iteration and thus perfoms the prefix sum. As expected, we have n-1 adds with a total asymtotic runtime of O(n). Very strait forward, but critically, this is a sequencial operation with no parallelization. Although, for relatively small inputs this is significantly faster than the more complex GPU algorithms we implement, as our input grows this becomes much slower. 

$\underline{Naive Parallel Scan}$

If we expand out each sum in our sequenctial sum, we see that we can break down that entire thing into independent pairs of sums. Then, once again, we can break those down into pairs of sums, and so on untill finally we have just one value left. The result looks like a binary tree where at each level we do half the number of adds of the previous level. Note also that at each level, we have actually already completed a sum for some of the outputs. The advantage of doing things this way is that each individual sum in a layer doesn't rely on the others, and can thus be computed in parallel. In my implementation, since we are reading and writing to the same array, I used a ping-pong array system for the output to avoid any incorrect adds. Overall, we get $O(nlog_2(n))$ adds with an asymtotic time of $O(log_2(n))$. Note, although we are doing many more adds, since they can be done in parallel, they can be completed faster 

![Naive Scan](img/NaiveScan_V1.png)

$\underline{Work-Efficient Parallel Scan}$

Although it might not be obvious at first we can actually cut out some of the addition operations.

![Up Sweep](img/UpSweeP_V1.png)

![Down Sweep](img/DownSweep_V1.png)

### Stream Compaction
The second part of this project implemented stream compaction of the input array by removing unwanted elements, which in this case was removing all 0 elements from our array of ints. Again, a sequencial algorithm seems obvious and possibly neessary. However, using this scan function we can parallelize how we find the end indices for each non-zero element as well as how we place those into our output array. Again, for comparisson, I implemented both a basic CPU version and this parallelized version.

$\underline{CPU Stream Compaction}$

As expected, this was implemented using a for loop and only adding elements to our new array if that elements is not 0. In the end we get the compacted version of our input array. Similarlly to scan, we are sequentially going through each element resulting in an asymtotic run time of $O(n)$. For the same reasons, this is better on small inputs, but won't scale as well as the parallel algorithm.

![CPU Stream Compaction](img/NaiveCompact_V1.png)

$\underline{GPU Stream Compaction}$

To parallelize this task, let's first think about what information we need to find. First, we need to know which elements of our input are good and which should be removed. This is obviously parallel since for each element we are doing the same check. A simple kernel which checks the element in index i and stores weather or not it is acceptable in index i of a boolean array can quickly do this. Next, now that we know which elements we should keep and their indices, we need to figure out which index in our out array should hold that element. This is the 'shift' step where after removing an element, we shift the entire array down to fill in the gap. To find this in parallel, we can actually reuse our scan algorithm. If we format our boolean array as 0s and 1s, performing scan on this algorithm will give us the exact indices we are looking for. For each element we want to keep, the scan result for that element's index will be the number of good elements that come before it since we are summing up all the 1's. Thus, we now know which elements we should keep, and their desired index in our output array. We can then read, in parallel, both our boolean array and our index array and only put the good elements into our output at the correct spots. As for the asymtotics of this algorithm, the most expensive step is actually doing the scan, which if we use our work efficient parallel scan, we get a time of $O(n)$. 

![GPU Stream Compaction](img/ParallelCompact_V1.png)

### A Quick Note On Groups
As mentioned earlier, with the Scan algorithm, and thus Stream Compaction aswell, we are simply given an array of elements and some operation that can be used on those elements. If we look further into the algorithm however, we see that two more conditions need to be met: our operation must be associative and there must be an identity element. This is precicely the axioms needed for a group! Although in practice we would need to do a bit more work and memory managment if our group element isn't a simple number that can be stored in an array, the algorithm will still work as expected. There is just one other thing we need to be careful of when implementing Down Sweep in the Work Efficient Scan: Commutativity. For many groups, including our intager addition we implemented, our group is abelian, which means A+B = B+A as you would expect. However this is not the case for all groups. For example, matrix multiplications is not-commutative and thus we need to be extra cearful when multiplying to ensure we maintain the ordering of or initial array. As presented in the pictures above, when doing the swap/combine step of two elements, doing Left * Right will actually break the order of our input array. However, if we do Right * Left at each step, we see the ordering gets maintained, meaning that we can still use these efficient methods for non-abelian groups. 

---
## Results

--- 

## Performance Analysis

---
## References
- [Nvidia Developer GPU Gems 3 ](https://developer.nvidia.com/gpugems/gpugems3/part-vi-gpu-computing/chapter-39-parallel-prefix-sum-scan-cuda)