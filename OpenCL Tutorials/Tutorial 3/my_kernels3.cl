//reduce using local memory (so called privatisation)
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

__kernel void min_val(__global const float* A, __global float* B, __local float* scratch) {
	int id = get_global_id(0);
	int lid = get_local_id(0);
	int N = get_local_size(0);

	//cache all N values from global memory to local memory
	scratch[lid] = A[id];

	barrier(CLK_LOCAL_MEM_FENCE);//wait for all local threads to finish copying from global to local memory

	for (int i = 1; i < N; i *= 2) {

		if (!(lid % (i * 2)) && ((lid + i) < N)) 
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

		if (!(lid % (i * 2)) && ((lid + i) < N)) 
			if (scratch[lid] < scratch[lid + i]) { 
			
				scratch[lid] = scratch[lid + i];
				
				}
		barrier(CLK_LOCAL_MEM_FENCE);
	}

	//copy the cache to output array
	B[id] = scratch[lid];
}

__kernel void variance(__global const float* A, __global float* B, __local float* scratch, __global const float* average) {
	int id = get_global_id(0);
	int lid = get_local_id(0);
	int N = get_local_size(0);

	//cache all N values from global memory to local memory
	scratch[lid] = ((A[id] - average[0]) * (A[id] - average[0]));

	barrier(CLK_LOCAL_MEM_FENCE);//wait for all local threads to finish copying from global to local memory

		for (int i = 1; i < N; i *= 2) {
		if (!(lid % (i * 2)) && ((lid + i) < N)) 
			scratch[lid] += scratch[lid + i];

		barrier(CLK_LOCAL_MEM_FENCE);
	}

	//copy the cache to output array
	B[id] = scratch[lid];
}

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

