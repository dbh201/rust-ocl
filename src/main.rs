use opencl3::platform::Platform;
use opencl3::context::Context;
use opencl3::device::Device;
use opencl3::program::Program;

use opencl3::platform::get_platforms;

use opencl3::device::CL_DEVICE_TYPE_CPU;
use opencl3::device::CL_DEVICE_TYPE_GPU;

use opencl3::memory::*;
use opencl3::command_queue::*;
use opencl3::kernel::*;

use md5::compute;
use std::os::raw::c_void;

fn main() {
    let platforms = get_platforms().unwrap();
    for (i, v) in platforms.iter().enumerate() {
        println!("Platform {}:",i);
        println!("  Name: {}",v.name().unwrap());
        println!("  Vendor: {}",v.vendor().unwrap());
        println!("  Version: {}",v.version().unwrap());
        println!("  ---DETAILS---");
        print_devices(v);
    }
    println!("There should be logic here, but for now platform and device is hardcoded to my specific machine.");
    println!("Creating context from GPU device 0 on platform 1...");

    let gpus = platforms[1].get_devices(CL_DEVICE_TYPE_GPU).expect("Couldn't get GPU devices!");
    let device = Device::new(gpus[0]);
    let ctx = Context::from_device(&device).expect("Couldn't create context!");
    println!("Generating test data ...");
    let mut data_generator: [u8;16] = [0;16];
    for i in 0..16 {
        data_generator[i] = rand::random::<u8>();
    }
    let mut data1 = data_generator;
    let mut data_generator: [u8;16] = [0;16];
    for i in 0..16 {
        data_generator[i] = rand::random::<u8>();
    }
    let mut data2 = data_generator;

    let mut datao: [u8;16] = [0;16];
    let src = "
void kernel simple_add(global const int* A, global const int* B, global int* C) {
   C[get_global_id(0)]=A[get_global_id(0)]+B[get_global_id(0)]; 
}";
    let program = Program::create_and_build_from_source( &ctx, &src, "" ).expect("Couldn't compile program!");
    let kernel = Kernel::create(&program, "simple_add").expect("Couldn't create kernel!");

    let buffer1;
    let buffer2;
    let output;
    let queue;
    unsafe {
        println!("UNSAFE");
        println!("Creating device buffers...");
        buffer1 = Buffer::<u8>::create(&ctx,CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,16,data1.as_mut_ptr() as *mut c_void).expect("Couldn't allocate memory for buffer 1");
        buffer2 = Buffer::<u8>::create(&ctx,CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,16,data2.as_mut_ptr() as *mut c_void).expect("Couldn't allocate memory for buffer 2");
        output = Buffer::<u8>::create(&ctx,CL_MEM_WRITE_ONLY | CL_MEM_USE_HOST_PTR,16,datao.as_mut_ptr() as *mut c_void).expect("Couldn't allocate memory for output");
        println!("Creating queue...");
        queue = CommandQueue::create_with_properties(&ctx,device.id(),0,0).expect("Couldn't create queue!");
        println!("Performing simple add...");
        let event = ExecuteKernel::new(&kernel).set_arg(&buffer1).set_arg(&buffer2).set_arg(&output).set_global_work_size(16).enqueue_nd_range(&queue).expect("Execution failed.");
        event.wait().expect("Event failed.");
        println!("SAFE");
    }
    println!("{:?}",data1);
    println!("{:?}",data2);
    println!("{:?}",datao);
}

fn print_devices(v: &Platform) {
    let cpus = v.get_devices(CL_DEVICE_TYPE_CPU).expect("Couldn't get CPU devices!");
    let gpus = v.get_devices(CL_DEVICE_TYPE_GPU).expect("Couldn't get GPU devices!");
    println!("  CPUS: {}",cpus.len());
    let mut j = 0;
    while j < cpus.len(){
        let device = Device::new(cpus[j]); 
        println!("    CPU {}: {} {}", j, device.vendor().unwrap(), device.name().unwrap());
        j+=1;
    }
    println!("  GPUS: {}",gpus.len());
    j = 0;
    while j < gpus.len() {
        let device = Device::new(gpus[j]); 
        println!("    GPU {}: {} {}", j, device.vendor().unwrap(), device.name().unwrap());
        j+=1;
    }
}
