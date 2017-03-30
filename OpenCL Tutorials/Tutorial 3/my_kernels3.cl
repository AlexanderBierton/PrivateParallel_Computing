//reduce using local memory (so called privatisation)
//mean Kernel using Local Memory
__kernel void mean_kernel(__global const float* A, __global float* B, __local float* scratch) {
	int id = get_global_id(0);
	int lid = get_local_id(0);
	int N = get_local_size(0);

	//cache all N values from global memory to local memory
	scratch[lid] = A[id];

	barrier(CLK_LOCAL_MEM_FENCE);//wait for all local threads to finish copying from global to local memory

	for (int i = 1; i < N; i *= 2) {
		if (!(lid % (i * 2)) && ((lid + i) < N)) 
			scratch[lid] += scratch[lid + i];

		barrier(CLK_LOCAL_MEM_FENCE);
	}

	//copy the cache to output array
	B[id] = scratch[lid];
}
//min_val Kernel using Local Memory
__kernel void min_val(__global const float* A, __global float* B, __local float* scratch) {
	int id = get_global_id(0);
	int lid = get_local_id(0);
	int N = get_local_size(0);

	//cache all N values from global memory to local memory
	scratch[lid] = A[id];

	barrier(CLK_LOCAL_MEM_FENCE);//wait for all local threads to finish copying from global to local memory

	for (int i = 1; i < N; i *= 2) {

		if (!(lid % (i * 2)) && ((lid + i) < N)) //this process allows for the kernel to find the lowest values in the work group
			if (scratch[lid] > scratch[lid + i]) { 
				scratch[lid] = scratch[lid + i];
				
				}
		barrier(CLK_LOCAL_MEM_FENCE);
	}

	//copy the cache to output array
	B[id] = scratch[lid];
}

__kernel void max_val(__global const float* A, __global float* B, __local float* scratch) {
	int id = get_global_id(0);
	int lid = get_local_id(0);
	int N = get_local_size(0);

	//cache all N values from global memory to local memory
	scratch[lid] = A[id];

	barrier(CLK_LOCAL_MEM_FENCE);//wait for all local threads to finish copying from global to local memory
	for (int i = 1; i < N; i *= 2) {

		if (!(lid % (i * 2)) && ((lid + i) < N)) //this function is the opposite to the min kernel allowing it to find the smallest number
			if (scratch[lid] < scratch[lid + i]) { 
			
				scratch[lid] = scratch[lid + i];
				
				}
		barrier(CLK_LOCAL_MEM_FENCE);
	}

	//copy the cache to output array
	B[id] = scratch[lid];
}

//variance kernel using local memory and also the mean buffer
__kernel void variance(__global const float* A, __global float* B, __local float* scratch, __global const float* average) {
	int id = get_global_id(0);
	int lid = get_local_id(0);
	int N = get_local_size(0);

	//cache all N values from global memory to local memory
	scratch[lid] = ((A[id] - average[0]) * (A[id] - average[0])); //this was done to allow for squaring the data and taking away the mean.

	barrier(CLK_LOCAL_MEM_FENCE);//wait for all local threads to finish copying from global to local memory

		for (int i = 1; i < N; i *= 2) {
		if (!(lid % (i * 2)) && ((lid + i) < N)) //adds together all the data in the work group to help find mean
			scratch[lid] += scratch[lid + i];

		barrier(CLK_LOCAL_MEM_FENCE);
	}

	//copy the cache to output array
	B[id] = scratch[lid];
}

//this parallel selection sort is basic and mostly inefficient in terms of sorting as it doesnt use local memory
__kernel void ParallelSelection(__global const float * in,__global float * out)
{
  int i = get_global_id(0); // current thread
  int n = get_global_size(0); // input size
  float iData = in[i];
  float iKey = iData;
  // Compute position of in[i] in output
  int pos = 0;
  for (int j=0;j<n;j++)
  {
    float jKey = in[j]; // broadcasted
    bool smaller = (jKey < iKey) || (jKey == iKey && j < i);  // in[j] < in[i] ?
    pos += (smaller)?1:0;
  }
  out[pos] = iData;
}

