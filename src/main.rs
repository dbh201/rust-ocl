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
use std::io::BufReader;
use std::io::prelude::*;

fn main() {
    const WORK_SIZE: usize = 1024*1024;
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

    println!("Retrieving OpenCL source ...");
    let srcFile = std::fs::File::open("md5.c").expect("Couldn't open source file.");
    let mut src = String::new();
    BufReader::new(srcFile).read_to_string(&mut src).expect("Couldn't load source file.");
    println!("Compiling program ...");
    let program = Program::create_and_build_from_source( &ctx, src.as_str(), "" ).expect("Couldn't compile program!");
    println!("Compiling kernel ...");
    let kernel = Kernel::create(&program, "getmd5").expect("Couldn't create kernel!");

    let mut sum: [[u8;16];WORK_SIZE] = [[0;16];WORK_SIZE];
    let mut block: [[u8;64];WORK_SIZE] = [[0;64];WORK_SIZE];
    let mut size: [usize;WORK_SIZE] = [0;WORK_SIZE];
    let mut expected: [[u8;16];WORK_SIZE] = [[0;16];WORK_SIZE];

    println!("Generating {} test blocks from 0-56 bytes ...", WORK_SIZE);
    for i in 0..WORK_SIZE {
        size[i] = rand::random::<usize>()%56;
        for j in 0..size[i] {
            block[i][j] = rand::random::<u8>();
        }
        //println!("JOB {}: {:02x?}",i,block[i]);
    }
    
    let outputSum;
    let inputBlock;
    let inputSize;
    let queue;
    let event;
    unsafe {
        println!("UNSAFE");
        println!("Creating device buffers...");
        outputSum = Buffer::<[u8;16]>::create(&ctx,CL_MEM_WRITE_ONLY | CL_MEM_USE_HOST_PTR,WORK_SIZE,sum.as_mut_ptr() as *mut c_void).expect("Couldn't allocate memory for targetSum");
        inputBlock = Buffer::<[u8;64]>::create(&ctx,CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR,WORK_SIZE,block.as_mut_ptr() as *mut c_void).expect("Couldn't allocate memory for outputRes");
        inputSize = Buffer::<usize>::create(&ctx,CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR,WORK_SIZE,size.as_mut_ptr() as *mut c_void).expect("Couldn't allocate memory for outputSize");
        println!("Creating queue...");
        queue = CommandQueue::create_with_properties(&ctx,device.id(),0,0).expect("Couldn't create queue!");
        println!("Dispatch mass MD5 computation event...");
        event = ExecuteKernel::new(&kernel).set_arg(&outputSum).set_arg(&inputBlock).set_arg(&inputSize).set_global_work_size(WORK_SIZE).enqueue_nd_range(&queue).expect("Execution failed.");
        println!("SAFE");
    }

    println!("Single-threaded CPU ...");
    for i in 0..WORK_SIZE {
        expected[i] = md5::compute(&block[i][0..size[i]]).into();
    }
    println!("Done single-threaded work.");
    event.wait().expect("Event waiting failed?");
    println!("Finished waiting on the GPU.");
    for i in 0..WORK_SIZE {
        for j in 0..16 {
            if sum[i][j] != expected[i][j] {
                println!("Job {} failed comparison:\nRetrieved: {:?}\n Expected: {:?}",i,sum[i],expected[i]);
                return;
            }
        }
    }
    println!("All tests passed!");
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
