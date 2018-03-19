#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "functions.h"

__global__ void defineCluster(Point* allPoints, int numOfPoints, Cluster* clusterCenters, int numOfClusters, int* allTerminationCondition)
{
	// pointIndex in allPoints = i
	int i = threadIdx.x + blockDim.x * blockIdx.x;
	double x, y;
	if (i < numOfPoints)
	{
		int lastIndexGiven;
		double oldDistFromCluster, distFromCluster;
		// Get point[i]'s position.
		x = allPoints[i].x;
		y = allPoints[i].y;
		// At the beginning of the iteration(iter) the point's center is it's old center, meaning it hasn't changed yet.
		allTerminationCondition[i] = 0;
		// Define a cluster to a point if it doesn't exists.
		if (allPoints[i].clusterIndex < 0)
		{
			allPoints[i].clusterIndex = 0;
			lastIndexGiven = -1;
		}
		else
			lastIndexGiven = allPoints[i].clusterIndex;
		// Calculate the distance from point[i] to it's own cluster center.
		oldDistFromCluster = sqrt(pow((clusterCenters[allPoints[i].clusterIndex].x - x), 2) + pow((clusterCenters[allPoints[i].clusterIndex].y - y), 2));
		// For each point[i] calculate the nearest cluster center.
		for (int j = 0; j < numOfClusters; j++)
		{
			// The new distance from point[i] to cluster[j].
			distFromCluster = sqrt(pow((clusterCenters[j].x - x), 2) + pow((clusterCenters[j].y - y), 2));
			/*If true:
			- Define new distance as "old"
			- Define last nearest cluster index. */
			if (distFromCluster < oldDistFromCluster)
			{
				oldDistFromCluster = distFromCluster;
				lastIndexGiven = j;
			}
		}
		// True only if point[i] gained a new cluster index
		if (lastIndexGiven != allPoints[i].clusterIndex)
		{
			allTerminationCondition[i] = 1;
			allPoints[i].clusterIndex = lastIndexGiven;
		}
	}
}

__global__ void updatePoint(Point* allPoints, int numOfPoints, double time)
{
	int i = threadIdx.x + blockDim.x * blockIdx.x;
	if (i < numOfPoints)
	{
		allPoints[i].x = allPoints[i].x0 + time*allPoints[i].vx;
		allPoints[i].y = allPoints[i].y0 + time*allPoints[i].vy;
	}
}

cudaError_t updatePointsPosCuda(Point* allPoints, int numOfPoints, double time)
{
	int numOfBlocks, maxThreads;

	Point* dev_allPoints;

	cudaError_t cudaStatus;
	cudaDeviceProp props;
	cudaGetDeviceProperties(&props, 0);
	maxThreads = props.maxThreadsPerBlock;
	numOfBlocks = 1 + numOfPoints / maxThreads;

	// Choose which GPU to run on, change this on a multi-GPU system.
	cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
		goto Error;
	}

	cudaStatus = cudaMalloc((void**)&dev_allPoints, numOfPoints * sizeof(Point));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc(dev_allPoints) failed!");
		goto Error;
	}

	cudaStatus = cudaMemcpy(dev_allPoints, allPoints, numOfPoints* sizeof(Point), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy(dev_terminationCondition-terminationCondition) failed!");
		goto Error;
	}

	updatePoint << < numOfBlocks, maxThreads >> > (dev_allPoints, numOfPoints, time);

	// Check for any errors launching the kernel
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "updatePoint launch failed: %s\n", cudaGetErrorString(cudaStatus));
		goto Error;
	}

	// cudaDeviceSynchronize waits for the kernel to finish, and returns any errors encountered during the launch.
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching updatePoint!\n", cudaStatus);
		goto Error;
	}

	cudaStatus = cudaMemcpy(allPoints, dev_allPoints, numOfPoints * sizeof(Point), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy(terminationCondition) failed!\n");
		goto Error;
	}

Error:

	cudaFree(dev_allPoints);

	return cudaStatus;
}

cudaError_t definePointClusterCuda(int* terminationCondition, Point* allPoints, int numOfPoints, Cluster* clusterCenters, int numOfClusters)
{
	int numOfBlocks, maxThreads;
	int* allTerminationCondition = (int*)malloc(numOfPoints * sizeof(int));
	int* dev_allTerminationCondition;
	Point* dev_allPoints;
	Cluster* dev_clusterCenters;

	cudaError_t cudaStatus;
	cudaDeviceProp props;
	cudaGetDeviceProperties(&props, 0);
	numOfBlocks = 1 + numOfPoints / props.maxThreadsPerBlock;
	maxThreads = props.maxThreadsPerBlock;

	// Choose which GPU to run on, change this on a multi-GPU system.
	cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
		goto Error;
	}

	cudaStatus = cudaMalloc((void**)&dev_allPoints, numOfPoints * sizeof(Point));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc(dev_allPoints) failed!");
		goto Error;
	}

	cudaStatus = cudaMalloc((void**)&dev_clusterCenters, numOfClusters * sizeof(Cluster));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc(dev_clusterCenters) failed!");
		goto Error;
	}

	cudaStatus = cudaMalloc((void**)&dev_allTerminationCondition, numOfPoints * sizeof(int));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc(dev_terminationQuality) failed!");
		goto Error;
	}

	cudaStatus = cudaMemcpy(dev_allPoints, allPoints, numOfPoints * sizeof(Point), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy(dev_allPoints-allPoints) failed!");
		goto Error;
	}

	cudaStatus = cudaMemcpy(dev_clusterCenters, clusterCenters, numOfClusters * sizeof(Cluster), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy(dev_clusterCenters-clusterCenters) failed!");
		goto Error;
	}

	defineCluster << < numOfBlocks, maxThreads >> > (dev_allPoints, numOfPoints, dev_clusterCenters, numOfClusters, dev_allTerminationCondition);

	// Check for any errors launching the kernel
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "defineCenter launch failed: %s\n", cudaGetErrorString(cudaStatus));
		goto Error;
	}

	// cudaDeviceSynchronize waits for the kernel to finish, and returns any errors encountered during the launch.
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching defineCenter!\n", cudaStatus);
		goto Error;
	}

	//Copy to Host
	cudaStatus = cudaMemcpy(allPoints, dev_allPoints, numOfPoints * sizeof(Point), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy(allPoints) failed!\n");
		goto Error;
	}
	//Copy to Host
	cudaStatus = cudaMemcpy(clusterCenters, dev_clusterCenters, numOfClusters * sizeof(Cluster), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy(clusterCenters) failed!\n");
		goto Error;
	}
	//Copy to Host
	cudaStatus = cudaMemcpy(allTerminationCondition, dev_allTerminationCondition, numOfPoints * sizeof(int), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy(clusterCenters) failed!\n");
		goto Error;
	}
	for (int i = 0; i < numOfPoints; i++)
		*terminationCondition |= allTerminationCondition[i];

Error:
	free(allTerminationCondition);
	cudaFree(dev_clusterCenters);
	cudaFree(dev_allPoints);
	cudaFree(dev_allTerminationCondition);

	return cudaStatus;
}
