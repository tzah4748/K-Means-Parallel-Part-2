# Parallel implementation of K-Means
Documentation

Final project

Course 10324, Parallel and Distributed Computation

2017 FALL Semester

**Introduction:**

This documentation is the final project assignment as part of &quot;Parallel and Distributed Computation (10324)&quot; course.

The assignment was to implement and parallelize the K-Means algorithm.
My solution is based on the **&quot;Simplified K-Means algorithm&quot;** given to us in the assignment file.

**How did I parallelize?**

The solution was parallelized with MPI, OpenMP and CUDA.
Parallelization Steps:
(P – Number of Processes in MPI)
(N – Number of Points)

1. First, each process gets his equal share of points.
Process 0 gets to deal with the remaining (N % P) points.
2. Each process uses his cuda device to define each point&#39;s cluster based on their distances.
3. Each process uses OpenMP technology to add each point to its corresponding cluster.
4. Process 0 gets the gathered information about the points and clusters back from all P processes.
5. Process 0 send all the gathered points to his cuda device to update their current location in time.
6. Process 0 uses OpenMP technology to calculate both the diameters of the clusters and the quality of clusters found.

**Solution Rational:**

- At first, when I solved the problem I tried to do as many actions as possible in the process&#39;s cuda device, alas I found out that any calculation that holds the GPU&#39;s threads for too long are not possible.
- So, I decided that the cuda device will deal with simple calculations rather than dealing with a large number of iterations for each thread.
A simple example might be assigning a point its current location, this action is simple, each thread needs to make a simple multiplication and its done.
- Large calculations that can take a bit of time were made with OpenMP.

**Complexity:**

The general case&#39;s complexity for my solution is:
O (N \* K \* I \* (T/DT))
N – Number of points.
K – Number of clusters to find.
I – Number of iterations in each delta time (DT) iteration.
T/DT – Number of time iterations.

**Project Prerequisites:**

- MPI – MPICH2 Installed.
Notes: Make sure you include the MPI &quot;lib&quot; and &quot;include&quot; folders as part of your project properties.
- OpenMP – Enable Open MP Support to your project.
- CUDA Installed.
Notes: It is very recommended to open an empty CUDA project and to copy the files of my project into it, I&#39;ve used CUDA 9.1 (should work just fine on other versions as well).