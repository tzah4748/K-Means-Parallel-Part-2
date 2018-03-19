#include <mpi.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <omp.h>
#include "cuda_runtime.h"

typedef struct cluster_t
{
	double x = 0;
	double y = 0;
	double xSum = 0;
	double ySum = 0;
	double diameter = 0;
	int pointsInCluster = 0;
} Cluster;

typedef struct point_t
{
	double x0 = 0;
	double y0 = 0;
	double x = 0;
	double y = 0;
	double vx = 0;
	double vy = 0;
	int clusterIndex = -1;
	int clusterChanged;
} Point;

MPI_Datatype createPointDataTypeMPI();

MPI_Datatype createClusterDataTypeMPI();

void checkFile(FILE* f);

Point* getPointsFromFile(const char* inputFile, int* numOfPoints, int* numOfClusters, double* timeInterval, double* deltaTime, int* maxIterations, double* qualityMeasure);

Cluster* initializeClusters(Point* allPoints, int numOfClusters);

void bcastInitialsMPI(int* myNumOfPoints, int* numOfClusters, double* timeInterval, double* deltaTime, int* maxIterations);

cudaError_t updatePointsPosCuda(Point* allPoints, int numOfPoints, double time);
void updatePointsPos(Point* allPoints, int numOfPoints, double time); //Created for debugging.

void scatterInPlace(void* send_data, int send_count, MPI_Datatype send_datatype, void* recv_data, int recv_count, MPI_Datatype recv_datatype, int root, MPI_Comm communicator, int myid);

cudaError_t definePointClusterCuda(int* terminationCondition, Point* allPoints, int numOfPoints, Cluster* clusterCenters, int numOfClusters);
int definePointCenter(int startIndex, int endIndex, Point* allPoints, double time, Cluster* clusterCenters, int numOfClusters);

void addPointToCluster(Point* allPoints, int numOfPoints, Cluster* clusterCenters, int numOfClusters);

void gatherInPlace(void* send_data, int send_count, MPI_Datatype send_datatype, void* recv_data, int recv_count, MPI_Datatype recv_datatype, int root, MPI_Comm communicator, int myid);

void joinClusters(int iter, Cluster* clusterCenters, int numOfClusters, int myid, int numprocs, MPI_Datatype MPI_CLUSTER);

void recvTerminationCondition(int myid, int numprocs, int* terminationCondition);

void recalculateClusters(Cluster* clusterCenters, int numOfClusters);

double calculateDiameter(Point* allPoints, int numOfPoints, int clusterIndex);

double calculateQM(Cluster* clusterCenters, int numOfClusters);

void writeResultsToFile(const char* outputPath, double time, int iter, double terminationQuality, Cluster* clusterCenters, int numOfClusters);

void createRandomPointsFile(const char* inputFile);

