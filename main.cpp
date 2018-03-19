#include "functions.h"

#define MASTER 0

int main(int argc, char *argv[])
{
	// MPI Initialize Functions
	int namelen, numprocs, myid;
	char processor_name[MPI_MAX_PROCESSOR_NAME];
	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &myid);
	MPI_Comm_size(MPI_COMM_WORLD, &numprocs);
	MPI_Get_processor_name(processor_name, &namelen);
	MPI_Status status;

	// Initials and Variables
	int numOfPoints, numOfClusters, maxIterations, myNumOfPoints, iter,
		remainderPoints = 0, terminationCondition = false, finishIteration = false, qualityReached = false;
	double deltaTime, timeInterval, qualityMeasure, time, terminationQuality, t1, t2;
	const char* inputFile = "C:\\Users\\t4zm4\\Desktop\\Afeka\\3rd Year\\1st Semester\\Parallel Distributed Computing\\4 - finalProject_K-Means\\Parallel_K-Means\\Parallel_K-Means\\input.txt";
	const char* outputFile = "C:\\Users\\t4zm4\\Desktop\\Afeka\\3rd Year\\1st Semester\\Parallel Distributed Computing\\4 - finalProject_K-Means\\Parallel_K-Means\\Parallel_K-Means\\output.txt";
	Point *allPoints;
	Cluster *clusterCenters;

	/*//Create a random input file
	if (myid == 0)
		createRandomPointsFile(inputFile);
	*/

	// Create MPI Types for MPI Functions.
	MPI_Datatype MPI_POINT = createPointDataTypeMPI();
	MPI_Datatype MPI_CLUSTER = createClusterDataTypeMPI();

	// Read input file and initialize master process's arrays.
	if (myid == MASTER)
	{
		allPoints = getPointsFromFile(inputFile, &numOfPoints, &numOfClusters, &timeInterval, &deltaTime, &maxIterations, &qualityMeasure);
		// Devide points to each process
		myNumOfPoints = numOfPoints / numprocs;
		// Remainder of devided points belongs to master process
		remainderPoints = numOfPoints % numprocs;
		clusterCenters = initializeClusters(allPoints, numOfClusters);
	}
	bcastInitialsMPI(&myNumOfPoints, &numOfClusters, &timeInterval, &deltaTime, &maxIterations);
	if (myid != 0)
	{
		allPoints = (Point*)calloc(myNumOfPoints, sizeof(Point));
		clusterCenters = (Cluster*)calloc(numOfClusters, sizeof(Cluster));
	}
	t1 = MPI_Wtime();
	for (time = 0; time < timeInterval; time += deltaTime)
	{
		if (myid == MASTER)
			updatePointsPosCuda(allPoints, numOfPoints, time);
		for (iter = 0; iter < maxIterations; iter++)
		{
			// Scatter the points: Every mpi process gets his equal share of points into his allPoints array.
			scatterInPlace(allPoints, myNumOfPoints, MPI_POINT, allPoints, myNumOfPoints, MPI_POINT, MASTER, MPI_COMM_WORLD, myid);
			// Broadcast the initial clusters to all mpi processes.
			MPI_Bcast(clusterCenters, numOfClusters, MPI_CLUSTER, MASTER, MPI_COMM_WORLD);
			definePointClusterCuda(&terminationCondition, allPoints, myNumOfPoints, clusterCenters, numOfClusters);
			addPointToCluster(allPoints, myNumOfPoints, clusterCenters, numOfClusters);
			// In a specific case where the number of points isn’t divisible by the number of processes.
			if (myid == 0)
				// WITHOUT CUDA: the amount of remainder points may be small, meaning there is no need for cuda's computing power
				// the allocation of gpu and memory copy will just take more time.
				terminationCondition |= definePointCenter((numOfPoints - remainderPoints), numOfPoints, allPoints, time, clusterCenters, numOfClusters);
			// Gather the points: Every mpi process send back his part of allPoints array back to MASTER(root)'s allPoints array
			gatherInPlace(allPoints, myNumOfPoints, MPI_POINT, allPoints, myNumOfPoints, MPI_POINT, MASTER, MPI_COMM_WORLD, myid);
			// Join clusters info (of every mpi process) into MASTER's cluster array (xSum, ySum, pointInCluster).
			joinClusters(iter, clusterCenters, numOfClusters, myid, numprocs, MPI_CLUSTER);
			// Recive the termination condition flag from all processes
			recvTerminationCondition(myid, numprocs, &terminationCondition);
			// False Positive - the flag is false if all points stayed in their clusters
			if ((!terminationCondition) & (myid == MASTER))
				finishIteration = true;
			// Broadcast the finishIteration flag to break\continue iterations of all processes
			MPI_Bcast(&finishIteration, 1, MPI_INT, MASTER, MPI_COMM_WORLD);
			if (finishIteration)
				break;
			terminationCondition = false;
			// Calculate cluster's new center and reset some info (xSum, ySum, pointInCluster).
			if (myid == MASTER)
				recalculateClusters(clusterCenters, numOfClusters);
		}
		// Check if the quality of current clusters is enough to stop time iterations loop.
		if (myid == MASTER)
		{
			#pragma omp parallel for
			for (int j = 0; j < numOfClusters; j++)
				// Calculate each cluster's diameter.
				clusterCenters[j].diameter = calculateDiameter(allPoints, numOfPoints, j);
			// Get the current Quality Measure of the given clusters.
			terminationQuality = calculateQM(clusterCenters, numOfClusters);
			if (terminationQuality <= qualityMeasure)
				qualityReached = true;
		}
		// Broadcast the qualityReached flag to all mpi processes
		MPI_Bcast(&qualityReached, 1, MPI_INT, MASTER, MPI_COMM_WORLD);
		if (qualityReached)
			break;
		finishIteration = false;
	}
	t2 = MPI_Wtime();
	if (myid == MASTER)
	{
		writeResultsToFile(outputFile, time, iter, terminationQuality, clusterCenters, numOfClusters);
		printf("Success!\n", outputFile);
		printf("K-Means Time: %f\n", (t2 - t1));
	}
	// Free memory allocations.
	free(allPoints);
	free(clusterCenters);
	MPI_Finalize();
	return 0;
}

MPI_Datatype createPointDataTypeMPI()
{
	MPI_Datatype PointType[8] = { MPI_DOUBLE, MPI_DOUBLE, MPI_DOUBLE, MPI_DOUBLE, MPI_DOUBLE, MPI_DOUBLE, MPI_INT, MPI_INT };
	MPI_Datatype MPI_POINT;
	int blocklen[8] = { 1, 1, 1, 1, 1, 1, 1, 1 };
	MPI_Aint disp[8];

	disp[0] = offsetof(Point, x0);
	disp[1] = offsetof(Point, y0);
	disp[2] = offsetof(Point, x);
	disp[3] = offsetof(Point, y);
	disp[4] = offsetof(Point, vx);
	disp[5] = offsetof(Point, vy);
	disp[6] = offsetof(Point, clusterIndex);
	disp[7] = offsetof(Point, clusterChanged);
	MPI_Type_create_struct(8, blocklen, disp, PointType, &MPI_POINT);
	MPI_Type_commit(&MPI_POINT);
	return MPI_POINT;
}

MPI_Datatype createClusterDataTypeMPI()
{
	MPI_Datatype ClusterType[6] = { MPI_DOUBLE, MPI_DOUBLE, MPI_DOUBLE, MPI_DOUBLE, MPI_DOUBLE, MPI_INT };
	MPI_Datatype MPI_CLUSTER;
	int blocklen[6] = { 1, 1, 1, 1, 1, 1 };
	MPI_Aint disp[6];

	disp[0] = offsetof(Cluster, x);
	disp[1] = offsetof(Cluster, y);
	disp[2] = offsetof(Cluster, xSum);
	disp[3] = offsetof(Cluster, ySum);
	disp[4] = offsetof(Cluster, diameter);
	disp[5] = offsetof(Cluster, pointsInCluster);
	MPI_Type_create_struct(6, blocklen, disp, ClusterType, &MPI_CLUSTER);
	MPI_Type_commit(&MPI_CLUSTER);
	return MPI_CLUSTER;
}

void checkFile(FILE* f)
{
	if (!f)
	{
		printf("Failed opening the file. Exiting!\n");
		exit(1);
	}
}

Point* getPointsFromFile(const char* inputFile, int* numOfPoints, int* numOfClusters, double* timeInterval, double* deltaTime, int* maxIterations, double* qualityMeasure)
{
	FILE* f;
	fopen_s(&f, inputFile, "r");
	checkFile(f);
	fscanf_s(f, "%d %d %lf %lf %d %lf", numOfPoints, numOfClusters, timeInterval, deltaTime, maxIterations, qualityMeasure);
	Point* allPoints = (Point*)calloc(*numOfPoints, sizeof(Point));
	int i;
	for (i = 0; i < *numOfPoints; i++)
		fscanf_s(f, "%lf %lf %lf %lf", &allPoints[i].x0, &allPoints[i].y0, &allPoints[i].vx, &allPoints[i].vy);
	fclose(f);
	return allPoints;
}

Cluster* initializeClusters(Point* allPoints, int numOfClusters)
{
	Cluster* clusterCenters = (Cluster*)calloc(numOfClusters, sizeof(Cluster));
	for (int i = 0; i < numOfClusters; i++)
	{
		clusterCenters[i].x = allPoints[i].x0;
		clusterCenters[i].y = allPoints[i].y0;
	}
	return clusterCenters;
}

void bcastInitialsMPI(int* myNumOfPoints, int* numOfClusters, double* timeInterval, double* deltaTime, int* maxIterations)
{
	MPI_Bcast(myNumOfPoints, 1, MPI_INT, MASTER, MPI_COMM_WORLD);
	MPI_Bcast(numOfClusters, 1, MPI_INT, MASTER, MPI_COMM_WORLD);
	MPI_Bcast(timeInterval, 1, MPI_DOUBLE, MASTER, MPI_COMM_WORLD);
	MPI_Bcast(deltaTime, 1, MPI_DOUBLE, MASTER, MPI_COMM_WORLD);
	MPI_Bcast(maxIterations, 1, MPI_INT, MASTER, MPI_COMM_WORLD);
}

void updatePointsPos(Point* allPoints, int numOfPoints, double time)
{
#pragma omp parallel for
	for (int i = 0; i < numOfPoints; i++)
	{
		allPoints[i].x = allPoints[i].x0 + time*allPoints[i].vx;
		allPoints[i].y = allPoints[i].y0 + time*allPoints[i].vy;
	}
}

void scatterInPlace(void* send_data, int send_count, MPI_Datatype send_datatype, void* recv_data, int recv_count, MPI_Datatype recv_datatype, int root, MPI_Comm communicator, int myid)
{
	if (myid == root)
		MPI_Scatter(send_data, send_count, send_datatype, MPI_IN_PLACE, recv_count, recv_datatype, root, communicator);
	else
		MPI_Scatter(NULL, send_count, send_datatype, recv_data, recv_count, recv_datatype, root, communicator);
}

int definePointCenter(int startIndex, int endIndex, Point* allPoints, double time, Cluster* clusterCenters, int numOfClusters)
{
	double x, y;
	int terminationCondition = false;
	double oldDistFromCluster, distFromCluster;
	for (int i = startIndex; i < endIndex; i++)
	{
		/*Get point[i]'s Position*/
		x = allPoints[i].x;
		y = allPoints[i].y;
		/*At the beginning of the iteration(iter) the point's center is it's old center, meaning it hasn't changed yet*/
		allPoints[i].clusterChanged = false;
		/*Define a center to a point if it doesn't exists*/
		if (allPoints[i].clusterIndex < 0)
			allPoints[i].clusterIndex = 0;
		/*Calculate the distance from point[i] to it's own cluster center*/
		oldDistFromCluster = sqrt(pow((clusterCenters[allPoints[i].clusterIndex].x - x), 2) + pow((clusterCenters[allPoints[i].clusterIndex].y - y), 2));
		/*For each point[i] calculate a new cluster center*/
		for (int j = 0; j < numOfClusters; j++)
		{
			/*The new distance from point[i] to cluster[j]*/
			distFromCluster = sqrt(pow((clusterCenters[j].x - x), 2) + pow((clusterCenters[j].y - y), 2));
			/*If true:
			- Define new distance as "old"
			- Define new center to the point[i]
			- Point[i]'s center has now changed.*/
			if (distFromCluster < oldDistFromCluster)
			{
				oldDistFromCluster = distFromCluster;
				allPoints[i].clusterIndex = j;
				allPoints[i].clusterChanged = true;
			}
		}
		/*After defining a point to its center, add the point to the center
		- This simple logic works due to the fact a cluster's new center is an average of all points in the cluster*/
		clusterCenters[allPoints[i].clusterIndex].xSum += x;
		clusterCenters[allPoints[i].clusterIndex].ySum += y;
		clusterCenters[allPoints[i].clusterIndex].pointsInCluster += 1;
		/*Termination condition/flag will stay true as long as at least one point[i] defined a new center*/
		terminationCondition |= allPoints[i].clusterChanged;
	}
	return terminationCondition;
}

void addPointToCluster(Point* allPoints, int numOfPoints, Cluster* clusterCenters, int numOfClusters)
{
	int numCores = omp_get_max_threads();
	omp_set_num_threads(numCores);
	Cluster* allClusterCenters = (Cluster*)calloc(numOfClusters * numCores, sizeof(Cluster));

#pragma omp parallel for
	for (int j = 0; j < numOfPoints; j++)
	{
		int tid = omp_get_thread_num();
		allClusterCenters[allPoints[j].clusterIndex + tid * numOfClusters].xSum += allPoints[j].x;
		allClusterCenters[allPoints[j].clusterIndex + tid * numOfClusters].ySum += allPoints[j].y;
		allClusterCenters[allPoints[j].clusterIndex + tid * numOfClusters].pointsInCluster += 1;
	}

#pragma omp parallel for
	for (int k = 0; k < numOfClusters; k++)
	{
		for (int i = 0; i < numCores; i++)
		{
			clusterCenters[k].xSum += allClusterCenters[k + i*numOfClusters].xSum;
			clusterCenters[k].ySum += allClusterCenters[k + i*numOfClusters].ySum;
			clusterCenters[k].pointsInCluster += allClusterCenters[k + i*numOfClusters].pointsInCluster;
		}
	}
	free(allClusterCenters);
}

void gatherInPlace(void* send_data, int send_count, MPI_Datatype send_datatype, void* recv_data, int recv_count, MPI_Datatype recv_datatype, int root, MPI_Comm communicator, int myid)
{
	if (myid == root)
		MPI_Gather(MPI_IN_PLACE, send_count, send_datatype, recv_data, recv_count, recv_datatype, root, communicator);
	else
		MPI_Gather(send_data, send_count, send_datatype, recv_data, recv_count, recv_datatype, root, communicator);
}

void joinClusters(int iter, Cluster* clusterCenters, int numOfClusters, int myid, int numprocs, MPI_Datatype MPI_CLUSTER)
{
	Cluster* allClusterCenters;
	if (myid == MASTER)
		allClusterCenters = (Cluster*)calloc(numOfClusters * numprocs, sizeof(Cluster));
	MPI_Gather(clusterCenters, numOfClusters, MPI_CLUSTER, allClusterCenters, numOfClusters, MPI_CLUSTER, MASTER, MPI_COMM_WORLD);
	if (myid == MASTER)
	{
		for (int i = 0; i < numOfClusters; i++)
		{
			for (int j = 1; j < numprocs; j++)
			{
				clusterCenters[i].xSum += allClusterCenters[(j*numOfClusters) + i].xSum;
				clusterCenters[i].ySum += allClusterCenters[(j*numOfClusters) + i].ySum;
				clusterCenters[i].pointsInCluster += allClusterCenters[(j*numOfClusters) + i].pointsInCluster;
			}
		}
	}
	if (myid == MASTER)
		free(allClusterCenters);
}

void recvTerminationCondition(int myid, int numprocs, int* terminationCondition)
{
	int returnedTerminationCondition;
	MPI_Status status;
	if (myid != MASTER)
		MPI_Send(terminationCondition, 1, MPI_INT, MASTER, 0, MPI_COMM_WORLD);
	else
		for (int i = 0; i < numprocs - 1; i++)
		{
			MPI_Recv(&returnedTerminationCondition, 1, MPI_INT, MPI_ANY_SOURCE, 0, MPI_COMM_WORLD, &status);
			*terminationCondition |= returnedTerminationCondition;
		}
}

void recalculateClusters(Cluster* clusterCenters, int numOfClusters)
{
#pragma omp parallel for
	for (int j = 0; j < numOfClusters; j++)
	{
		clusterCenters[j].x = clusterCenters[j].xSum / clusterCenters[j].pointsInCluster;
		clusterCenters[j].y = clusterCenters[j].ySum / clusterCenters[j].pointsInCluster;
		clusterCenters[j].xSum = 0;
		clusterCenters[j].ySum = 0;
		clusterCenters[j].pointsInCluster = 0;
	}
}

double calculateDiameter(Point* allPoints, int numOfPoints, int clusterIndex)
{
	double diameter = 0;
	double ret = 0;
	for (int i = 0; i < numOfPoints; i++)
	{
		Point a = allPoints[i];
		if (a.clusterIndex == clusterIndex)
		{
			for (int j = i + 1; j < numOfPoints; j++)
			{
				Point b = allPoints[j];
				if (b.clusterIndex == clusterIndex)
				{
					diameter = sqrt(pow((a.x - b.x), 2) + pow((a.y - b.y), 2));
					if (diameter > ret)
						ret = diameter;
				}
			}
		}
	}
	return ret;
}

double calculateQM(Cluster* clusterCenters, int numOfClusters)
{
	int i, j;
	double q = 0;
	double count = 0;
#pragma omp parallel for private(j) reduction(+ : q) reduction(+ : count)
	for (i = 0; i < numOfClusters; i++)
	{
		for (j = 0; j < numOfClusters; j++)
		{
			if (i != j)
			{
				q += clusterCenters[i].diameter / sqrt(pow((clusterCenters[i].x - clusterCenters[j].x), 2) + pow((clusterCenters[i].y - clusterCenters[j].y), 2));
				count += 1;
			}
		}
	}
	q = q / count;
	return q;
}

void writeResultsToFile(const char* outputFile, double time, int iter, double terminationQuality, Cluster* clusterCenters, int numOfClusters)
{
	FILE* f;
	fopen_s(&f, outputFile, "w");
	checkFile(f);
	fprintf(f, "First occurence at t = %f, iter = %d with q = %f\n\nCenter of the Clusters:\n\n", time, iter, terminationQuality);
	for (int i = 0; i < numOfClusters; i++)
	{
		fprintf(f, "%f %f\n\n", clusterCenters[i].x, clusterCenters[i].y);
	}
	fclose(f);
}

void printPoints(Point* p, int numOfPoints)
{
	for (int i = 0; i < numOfPoints; i++)
		printf("%0.2f %0.2f\n", p[i].x, p[i].y);
}

void createRandomPointsFile(const char* inputFile)
{
	FILE* f;
	fopen_s(&f, inputFile, "w");
	checkFile(f);
	srand((unsigned int)time(NULL));
	fputs("20000 4 30 0.1 2000 7.3\n", f);
	for (int i = 0; i < 20000; i++)
		fprintf(f, "%.2f %.2f %.2f %.2f\n", double((rand() % 20000) - 10000) / 100, double((rand() % 20000) - 10000) / 100, double((rand() % 40000) - 20000) / 100, double((rand() % 40000) - 20000) / 100);
	fclose(f);
}