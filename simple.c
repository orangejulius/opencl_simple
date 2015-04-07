#define PROGRAM_FILE "simple.cl"
#define KERNEL_FUNC "simple"

#include <stdio.h>
#include <stdlib.h>
#include <sys/types.h>

#include <CL/cl.h>

int main() {

   /* Host/device data structures */
   cl_platform_id platform;
   cl_device_id device;
   cl_context context;
   cl_command_queue queue;
   cl_int i, err;

   /* Program/kernel data structures */
   cl_program program;
   FILE *program_handle;
   char *program_buffer, *program_log;
   size_t program_size, log_size;
   cl_kernel kernel;

   /* Data and buffers */
   float result[1] = {0.0};
   cl_mem res_buff;
   size_t work_units_per_kernel;

   /* Identify a platform */
   err = clGetPlatformIDs(1, &platform, NULL);
   if(err < 0) {
      perror("Couldn't find any platforms");
      exit(1);
   }

   /* Access a device */
   err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL);
   if(err < 0) {
      perror("Couldn't find any devices");
      exit(1);
   }

   /* Create the context */
   context = clCreateContext(NULL, 1, &device, NULL, NULL, &err);
   if(err < 0) {
      perror("Couldn't create a context");
      exit(1);
   }

   /* Read program file and place content into buffer */
   program_handle = fopen(PROGRAM_FILE, "r");
   if(program_handle == NULL) {
      perror("Couldn't find the program file");
      exit(1);
   }
   fseek(program_handle, 0, SEEK_END);
   program_size = ftell(program_handle);
   rewind(program_handle);
   program_buffer = (char*)malloc(program_size + 1);
   program_buffer[program_size] = '\0';
   fread(program_buffer, sizeof(char), program_size, program_handle);
   fclose(program_handle);

   /* Create program from file */
   program = clCreateProgramWithSource(context, 1,
      (const char**)&program_buffer, &program_size, &err);
   if(err < 0) {
      perror("Couldn't create the program");
      exit(1);
   }
   //free(program_buffer);

   /* Build program */
   err = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
   if(err < 0) {

      /* Find size of log and print to std output */
      clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG,
            0, NULL, &log_size);
      program_log = (char*) malloc(log_size + 1);
      program_log[log_size] = '\0';
      clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG,
            log_size + 1, program_log, NULL);
      printf("%s\n", program_log);
      //free(program_log);
      exit(1);
   }

   /* Create kernel for the mat_vec_mult function */
   kernel = clCreateKernel(program, KERNEL_FUNC, &err);
   if(err < 0) {
      perror("Couldn't create the kernel");
      exit(1);
   }

   /* Create CL buffers to hold output data */
   res_buff = clCreateBuffer(context, CL_MEM_WRITE_ONLY,
      sizeof(float)*4, NULL, NULL);

   /* Create kernel arguments from the CL buffers */
   err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &res_buff);
   if(err < 0) {
      perror("Couldn't set the kernel argument");
      exit(1);
   }

   /* Create a CL command queue for the device*/
   queue = clCreateCommandQueue(context, device, 0, &err);
   if(err < 0) {
      perror("Couldn't create the command queue");
      exit(1);
   }

   /* Enqueue the command queue to the device */
   work_units_per_kernel = 4; /* 4 work-units per kernel */
   err = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &work_units_per_kernel,
      NULL, 0, NULL, NULL);
   if(err < 0) {
      perror("Couldn't enqueue the kernel execution command");
      exit(1);
   }

   /* Read the result */
   err = clEnqueueReadBuffer(queue, res_buff, CL_TRUE, 0, sizeof(float)*4,
      result, 0, NULL, NULL);
   if(err < 0) {
      perror("Couldn't enqueue the read buffer command");
      exit(1);
   }
   /* Test the result */
   if (result[0] == 1.0) {
	   printf("kernel successfully set result to 1.0\n");
   }
   else if (result[0] == 0.0) {
	   printf("kernel value stayed at default of 0.0, likely didn't run\n");
   } else {
	   printf("kernel set result to errornous value: %f\n", result[0]);
   }

   /* Deallocate resources */
   clReleaseMemObject(res_buff);
   clReleaseKernel(kernel);
   clReleaseCommandQueue(queue);
   clReleaseProgram(program);
   clReleaseContext(context);
}
