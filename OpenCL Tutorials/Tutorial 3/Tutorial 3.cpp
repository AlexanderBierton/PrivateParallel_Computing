#define CL_USE_DEPRECATED_OPENCL_1_2_APIS
#define __CL_ENABLE_EXCEPTIONS

#include <iostream>
#include <vector>
#include <fstream>
#include <sstream>

#ifdef __APPLE__
#include <OpenCL/cl.hpp>
#else
#include <CL/cl.hpp>
#endif

#include "Utils.h"
std::string contents;
std::vector<string> linevec;
std::vector<float> tempvec;

void print_help() {
	std::cerr << "Application usage:" << std::endl;

	std::cerr << "  -p : select platform " << std::endl;
	std::cerr << "  -d : select device" << std::endl;
	std::cerr << "  -l : list all platforms and devices" << std::endl;
	std::cerr << "  -h : print this message" << std::endl;
}

void aquire_info() {
	int colCount;
	
	for each (string let in linevec)
	{
		int i = 0;
		string temp;
		for each(char space in let) {
			if (space == ' ') {
				i++;
			}
			if (i == 5) {
				temp += space;
			}

		}
		tempvec.push_back(stof(temp));
	}
}

void readFile() {
	//Read in the text file	//	//	//	//	//	//	//	//
	std::ifstream infile("temp_lincolnshire.txt");
	std::string line;
	int colCount;
	while (std::getline(infile, line))
	{	
		linevec.push_back(line);
	}
	//std::cout << contents;
	//	//	//	//	//	//	//	//	//	//	//	//	//	//
}

int main(int argc, char **argv) {
	//Part 1 - handle command line options such as device selection, verbosity, etc.
	int platform_id = 0;
	int device_id = 0;

	for (int i = 1; i < argc; i++) {
		if ((strcmp(argv[i], "-p") == 0) && (i < (argc - 1))) { platform_id = atoi(argv[++i]); }
		else if ((strcmp(argv[i], "-d") == 0) && (i < (argc - 1))) { device_id = atoi(argv[++i]); }
		else if (strcmp(argv[i], "-l") == 0) { std::cout << ListPlatformsDevices() << std::endl; }
		else if (strcmp(argv[i], "-h") == 0) { print_help(); }
	}
	std::cout << "=====================================\n";
	std::cout << "Reading File..." <<std::endl; 

	readFile();
	aquire_info();
	int len = tempvec.size();

	std::cout << "File Read..." << std::endl;
	std::cout << "=====================================\n";

	//detect any potential exceptions
	try {
		//Part 2 - host operations
		//2.1 Select computing devices
		cl::Context context = GetContext(platform_id, device_id);

		//display the selected device
		std::cout << "Runinng on " << GetPlatformName(platform_id) << ", " << GetDeviceName(platform_id, device_id) << std::endl;

		//create a queue to which we will push commands for the device
		cl::CommandQueue queue(context, CL_QUEUE_PROFILING_ENABLE);

		//2.2 Load & build the device code
		cl::Program::Sources sources;

		AddSources(sources, "my_kernels3.cl");

		cl::Program program(context, sources);

		//build and debug the kernel code
		try {
			program.build();
		}
		catch (const cl::Error& err) {
			std::cout << "Build Status: " << program.getBuildInfo<CL_PROGRAM_BUILD_STATUS>(context.getInfo<CL_CONTEXT_DEVICES>()[0]) << std::endl;
			std::cout << "Build Options:\t" << program.getBuildInfo<CL_PROGRAM_BUILD_OPTIONS>(context.getInfo<CL_CONTEXT_DEVICES>()[0]) << std::endl;
			std::cout << "Build Log:\t " << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(context.getInfo<CL_CONTEXT_DEVICES>()[0]) << std::endl;
			throw err;
		}

		//Part 4 - memory allocation
		//host - input
		std::vector<float> A;//(10, 1);//allocate 10 elements with an initial value 1 - their sum is 10 so it should be easy to check the results!
		A = tempvec;
		//the following part adjusts the length of the input vector so it can be run for a specific workgroup size
		//if the total input length is divisible by the workgroup size
		//this makes the code more efficient
		size_t wg_size = 600;

		size_t padding_size = A.size() % wg_size;

		//if the input vector is not a multiple of the local_size
		//insert additional neutral elements (0 for addition) so that the total will not be affected
		if (padding_size) {
			//create an extra vector with neutral values
			std::vector<int> A_ext(wg_size-padding_size, 0);
			//append that extra vector to our input
			A.insert(A.end(), A_ext.begin(), A_ext.end());
		}

		size_t input_elements = A.size();//number of input elements
		size_t input_size = A.size()*sizeof(float);//size in bytes
		size_t nr_groups = input_elements / wg_size;

		//host - output
		std::vector<float> B(input_elements);
		size_t output_size = B.size()*sizeof(float);//size in bytes

		std::vector<float> C(input_elements);
		size_t output_sizeC = C.size() * sizeof(float);//size in bytes

		std::vector<float> D(input_elements);
		size_t output_sizeD = D.size() * sizeof(float);//size in bytes

		std::vector<float> E(input_elements);
		size_t output_sizeE = E.size() * sizeof(float);//size in bytes

		std::vector<float> F(input_elements);
		size_t output_sizeF = F.size() * sizeof(float);//size in bytes

		cl_ulong mean_time = 0;
		cl_ulong min_time = 0;
		cl_ulong max_time = 0;
		cl_ulong sd_time = 0;
		cl_ulong sort_time = 0;
		cl_ulong final_time = 0;


		//device - buffers
		cl::Buffer buffer_A(context, CL_MEM_READ_ONLY, input_size);
		cl::Buffer buffer_B(context, CL_MEM_READ_WRITE, output_size);
		cl::Buffer buffer_C(context, CL_MEM_READ_WRITE, output_size);
		cl::Buffer buffer_D(context, CL_MEM_READ_WRITE, output_size);
		cl::Buffer buffer_E(context, CL_MEM_READ_WRITE, output_size);
		cl::Buffer buffer_F(context, CL_MEM_READ_WRITE, output_size);
		cl::Buffer mean_buff(context, CL_MEM_READ_WRITE, sizeof(float));
		
		//Part 5 - device operations
		 
		//5.1 copy array A to and initialise other arrays on device memory
		queue.enqueueWriteBuffer(buffer_A, CL_TRUE, 0, input_size, &A[0]);
		queue.enqueueFillBuffer(buffer_B, 0, 0, output_size);//zero B buffer on device memory
		queue.enqueueFillBuffer(buffer_C, 0, 0, output_sizeC);
		queue.enqueueFillBuffer(buffer_D, 0, 0, output_sizeD);
		queue.enqueueFillBuffer(buffer_E, 0, 0, output_sizeE);
		queue.enqueueFillBuffer(buffer_F, 0, 0, output_sizeF);

		//5.2 Setup and execute all kernels (i.e. device code)
		cl::Kernel mean_kernel = cl::Kernel(program, "mean_kernel");
		mean_kernel.setArg(0, buffer_A);
		mean_kernel.setArg(1, buffer_B);
		mean_kernel.setArg(2, cl::Local(wg_size*sizeof(float)));//local memory size

		cl::Kernel min_kernel = cl::Kernel(program, "min_val");
		min_kernel.setArg(0, buffer_A);
		min_kernel.setArg(1, buffer_C);
		min_kernel.setArg(2, cl::Local(wg_size * sizeof(float)));//local memory size

		cl::Kernel max_kernel = cl::Kernel(program, "max_val");
		max_kernel.setArg(0, buffer_A);
		max_kernel.setArg(1, buffer_D);
		max_kernel.setArg(2, cl::Local(wg_size * sizeof(float)));//local memory size

		


		cl::Event profile_event;

		//call all kernels in a sequence
		queue.enqueueNDRangeKernel(mean_kernel, cl::NullRange, cl::NDRange(input_elements), cl::NDRange(wg_size), NULL, &profile_event);
		queue.enqueueReadBuffer(buffer_B, CL_TRUE, 0, output_size, &B[0], NULL, &profile_event);
		mean_time = profile_event.getProfilingInfo<CL_PROFILING_COMMAND_END>() - profile_event.getProfilingInfo<CL_PROFILING_COMMAND_START>();


		queue.enqueueNDRangeKernel(min_kernel, cl::NullRange, cl::NDRange(input_elements), cl::NDRange(wg_size), NULL, &profile_event);
		queue.enqueueReadBuffer(buffer_C, CL_TRUE, 0, output_sizeC, &C[0], NULL, &profile_event);
		min_time = profile_event.getProfilingInfo<CL_PROFILING_COMMAND_END>() - profile_event.getProfilingInfo<CL_PROFILING_COMMAND_START>();


		queue.enqueueNDRangeKernel(max_kernel, cl::NullRange, cl::NDRange(input_elements), cl::NDRange(wg_size), NULL, &profile_event);
		queue.enqueueReadBuffer(buffer_D, CL_TRUE, 0, output_sizeD, &D[0], NULL, &profile_event);
		max_time = profile_event.getProfilingInfo<CL_PROFILING_COMMAND_END>() - profile_event.getProfilingInfo<CL_PROFILING_COMMAND_START>();
		

		//5.3 Copy the result from device to host

		float overall = 0;

		//Go through each data to get overall result
		for (int i = 0; i < B.size(); i += wg_size) {
			overall += B[i];
		}

		float mean = overall / len;
		float mini = 0;
		float maxi = 0;

		for (int i = 0; i < C.size(); i += wg_size) {
			if (C[i] < mini) {
				mini = C[i];
			}
		}

		for (int i = 0; i < D.size(); i += wg_size) {
			if (D[i] > maxi) {
				maxi = D[i];
			}
		}

		queue.enqueueWriteBuffer(mean_buff, CL_TRUE, 0, sizeof(float), &mean);

		cl::Kernel var_kernel = cl::Kernel(program, "variance");
		var_kernel.setArg(0, buffer_A);
		var_kernel.setArg(1, buffer_E);
		var_kernel.setArg(2, cl::Local(wg_size * sizeof(float)));//local memory size
		var_kernel.setArg(3, mean_buff);//local memory size



		queue.enqueueNDRangeKernel(var_kernel, cl::NullRange, cl::NDRange(input_elements), cl::NDRange(wg_size), NULL, &profile_event);
		queue.enqueueReadBuffer(buffer_E, CL_TRUE, 0, output_sizeE, &E[0], NULL, &profile_event);
		sd_time = profile_event.getProfilingInfo<CL_PROFILING_COMMAND_END>() - profile_event.getProfilingInfo<CL_PROFILING_COMMAND_START>();

		

		

		float std_dev = 0;
		for (int i = 0; i < E.size(); i += wg_size) {
			std_dev += E[i];
		}
		std_dev = sqrt(std_dev / len);

		final_time = mean_time + min_time + max_time + sd_time;

		


		std::cout << "\n=====================================\n";
		
		std::cout << "Unsorted Data\n" <<std::endl;

		std::cout << "Mean: " << mean;

		std::cout << "\t\tSD: " << std_dev << std::endl;

		std::cout << "\nMin: " << mini;

		std::cout << "\t\tMax: " << maxi << std::endl;

		std::cout << "=====================================\n";

		std::cout<<"Kernel Information: " <<std::endl;

		std::cout << "\tMean Kernel Ex time[ns]: " << mean_time << std::endl;

		std::cout << "\tMin Kernel Ex time[ns]: " << min_time << std::endl;

		std::cout << "\tMax Kernel Ex time[ns]: " << max_time << std::endl;

		std::cout << "\tSD Kernel Ex time[ns]: " << sd_time << std::endl;

		std::cout << "\tOverall Kernel Execution time[ns]: " << final_time << std::endl;

		std::cout << "\n=====================================\n";

		std::cout << "Starting Sort...\n" << std::endl;


		cl::Kernel sort_kernel = cl::Kernel(program, "ParallelSelection");
		sort_kernel.setArg(0, buffer_A);
		sort_kernel.setArg(1, buffer_F);
		//sort_kernel.setArg(2, cl::Local(wg_size * sizeof(float)));//local memory size

		queue.enqueueNDRangeKernel(sort_kernel, cl::NullRange, cl::NDRange(input_elements), cl::NDRange(wg_size), NULL, &profile_event);
		queue.enqueueReadBuffer(buffer_F, CL_TRUE, 0, output_sizeF, &F[0], NULL, &profile_event);
		sort_time = profile_event.getProfilingInfo<CL_PROFILING_COMMAND_END>() - profile_event.getProfilingInfo<CL_PROFILING_COMMAND_START>();

		std::cout << "Finished Sort..." << std::endl;

		float median = ((F.size() - 1) /2);
		float loq = ((F.size() - 1) * 0.25);
		float upq = ((F.size() - 1) * 0.75);

		std::cout << "=====================================\n";

		std::cout << "Sorted Data\n" << std::endl;

		std::cout << "Sorted Min: " << F[0];

		std::cout << "\t\tSorted Max: " << F[F.size() - 1] << std::endl;

		std::cout << "\nMedian: " << F[median] << std::endl;

		std::cout << "\nUpper Quartile: " << F[upq];

		std::cout << "\tLower Quartile: " << F[loq] << std::endl;

		std::cout << "\n=====================================\n";

		std::cout << "Kernel Information: " << std::endl;
		
		std::cout << "\tSort Time: " << sort_time << std::endl;
		
		std::cout << "\t" << GetFullProfilingInfo(profile_event, ProfilingResolution::PROF_US) << endl;
		
		std::cout << "=====================================\n";



	}
	catch (cl::Error err) {
		std::cerr << "ERROR: " << err.what() << ", " << getErrorString(err.err()) << std::endl;
	}

	return 0;
}
