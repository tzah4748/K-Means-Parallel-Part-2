<h1>Parallel implementation of K-Means

Documentation</h1>


**Final project**  
**Course 10324, Parallel and Distributed Computation**
**2017 FALL Semester**

***Introduction:***

This documentation is the final project assignment as part of “Parallel
and Distributed Computation (10324)” course.

The assignment was to implement and parallelize the K-Means algorithm.  
My solution is based on the **“Simplified K-Means algorithm”** given to
us in the assignment file.

***Problem Definition:***
Given a set of points in **2**-dimensional space.

Initial position
**(x<sub>i</sub>, y<sub>i</sub>)** and velocity **(v<sub>xi</sub>,
> v<sub>yi</sub>)** are known for each point
> **P<sub>i</sub>**<sub>.</sub> Its position at the given time **t** can
> be calculated as follows:
>
> x<sub>i</sub>(t) = x<sub>i</sub> + t\*v<sub>xi</sub>
>
> y<sub>i</sub>(t) = y<sub>i</sub> + t\*v<sub>yi</sub>
>
> Implement simplified K-Means algorithm to find **K** clusters. Find a
> first occurrence during given time interval \[0, T\] when a system of
> **K** clusters has a Quality Measure **q** that is less than given
> value **QM**.
>
> ***Simplified K-Means algorithm:***

1.  Choose first **K** points as a cluster centers.

2.  Group points around the given cluster centers - for each point
    define a center that is most close to the point.

3.  Recalculate the cluster centers (average of all points in
    the cluster)

4.  Check the termination condition – no points move to other clusters
    or maximum iteration LIMIT was made.

5.  Repeat from 2 till the termination condition fulfills.

6.  Evaluate the Quality of the clusters found. The Quality is equal to
    an average of diameters of the cluster divided by distance to
    other clusters. For example, in case of k = 3 the quality is equal

> **q = (d<sub>1</sub>/D<sub>12</sub> + d<sub>1</sub>/D<sub>13</sub> +
> d<sub>2</sub>/D<sub>21</sub> + d<sub>2</sub>/D<sub>23</sub> +
> d<sub>3</sub>/D<sub>31</sub> + d<sub>3</sub>/D<sub>32</sub>) / 6**,
>
> where d<sub>i</sub> is a diameter of cluster **i** and D<sub>ij</sub>
> is a distance between centers of cluster **i** and cluster **j**.
>
> ***Input data and Output Result of the project:***
>
> You will be supplied with the following data

-   **N** - number of points.

-   **K** - number of clusters to find.

-   **LIMIT** – the maximum number of iterations for K-MEAN algorithm.

-   **QM** – quality measure to stop.

-   **T** – defines the end of time interval \[0, T\].

-   **dT** – defines moments t = n\*dT, n = {0, 1, 2, …, T/dT} for which
    calculate the clusters and the quality.

-   Coordinates and Velocities of all points.

> <span id="_Hlk509228352" class="anchor"></span>***Input File format***

The first line of the file contains **N K T dT LIMIT QM**. Next lines
are Initial Positions and Velocities of the points

*For example:*

5000 4 30 0.1 2000 7.3

2.3 4. 5 6. 55 -2.3

76.2 -3.56 50.0 12

…

45.23 20 -167.1 98

***Output File format***

The output file contains information on the found clusters with the
moment when the Quality Measure QM is reached for first time. For
example:

**First occurrence at t = 24.5 with q = 6.9**

**Centers of the clusters:**

1.123 34

-5.3 17.01

33.56 -23

14.1 98

> ***How did I parallelize?***
>
> The solution was parallelized with MPI, OpenMP and CUDA.  
> Parallelization Steps:  
> (P – Number of Processes in MPI)  
> (N – Number of Points)

1.  First, each process gets his equal share of points.  
    Process 0 gets to deal with the remaining (N % P) points.

2.  Each process uses his cuda device to define each point’s cluster
    based on their distances.

3.  Each process uses OpenMP technology to add each point to its
    corresponding cluster.

4.  Process 0 gets the gathered information about the points and
    clusters back from all P processes.

5.  Process 0 send all the gathered points to his cuda device to update
    their current location in time.

6.  Process 0 uses OpenMP technology to calculate both the diameters of
    the clusters and the quality of clusters found.

***Solution Rational:***

-   At first, when I solved the problem I tried to do as many actions as
    possible in the process’s cuda device, alas I found out that any
    calculation that holds the GPU’s threads for too long are
    not possible.

-   So, I decided that the cuda device will deal with simple
    calculations rather than dealing with a large number of iterations
    for each thread.  
    A simple example might be assigning a point its current location,
    this action is simple, each thread needs to make a simple
    multiplication and its done.

-   Large calculations that can take a bit of time were made
    with OpenMP.

***Complexity:***

The general case’s complexity for my solution is:  
O (N \* K \* I \* (T/DT))  
N – Number of points.  
K – Number of clusters to find.  
I – Number of iterations in each delta time (DT) iteration.  
T/DT – Number of time iterations.

***Project Prerequisites: ***

-   MPI – MPICH2 Installed.  
    Notes: Make sure you include the MPI “lib” and “include” folders as
    part of your project properties.

-   OpenMP – Enable Open MP Support to your project.

-   CUDA Installed.  
    Notes: It is very recommended to open an empty CUDA project and to
    copy the files of my project into it, I’ve used CUDA 9.1 (should
    work just fine on other versions as well).

