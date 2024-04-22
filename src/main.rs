extern crate vulkano;
extern crate winit;

use image::{RgbImage, Rgb};
use std::{intrinsics::size_of, io};

use vulkano::{
    buffer::{
        Buffer,
        BufferCreateInfo,
        BufferUsage,
    },
    pipeline::{
        Pipeline,
        PipelineShaderStageCreateInfo,
        PipelineBindPoint,
        ComputePipeline,
        PipelineLayout,
        layout::PipelineDescriptorSetLayoutCreateInfo,
        compute::ComputePipelineCreateInfo,
    },
    memory::allocator::{
        StandardMemoryAllocator,
        AllocationCreateInfo,
        MemoryTypeFilter,
        DeviceLayout,
    },
    descriptor_set::{
        allocator::StandardDescriptorSetAllocator,
        persistent::PersistentDescriptorSet,
        WriteDescriptorSet,
    },
    command_buffer::{
        allocator::StandardCommandBufferAllocator,
        CommandBufferUsage,
        AutoCommandBufferBuilder,
    },
    instance::{
        Instance,
        InstanceCreateInfo,
        InstanceCreateFlags,
    },
    device::{
        Device,
        DeviceCreateInfo,
        physical::PhysicalDeviceType,
        DeviceExtensions,
        Features,
        QueueCreateInfo,
        QueueFlags,
    },
    DeviceSize,
    sync,
    sync::GpuFuture,
    VulkanLibrary,
};

use std::sync::Arc;

const MINIMAL_FEATURES: Features = Features {
    geometry_shader: true,
    ..Features::empty()
};

fn main() {

    let mut input_string = String::new();
    io::stdin()
        .read_line(&mut input_string)
        .expect("Failed to read");

    // let factor = 8u32;
    let factor = input_string.trim().parse::<u32>().unwrap();

    let image_width = 2u32.pow(factor);
    let image_height = 2u32.pow(factor);
    let image_bytes = image_width * image_height * 3;

    // Boilerplate Initialization
    let library = VulkanLibrary::new().unwrap();
    let instance = Instance::new(
        library,
        InstanceCreateInfo {
            flags: InstanceCreateFlags::ENUMERATE_PORTABILITY,
            ..Default::default()
        },
        // InstanceCreateInfo::application_from_cargo_toml(),
    )
    .unwrap();

    let device_extensions = DeviceExtensions {
        khr_storage_buffer_storage_class: true,
        ..DeviceExtensions::empty()
    };

    // Getting the Device
    let (physical_device, queue_family_index) = instance
        .enumerate_physical_devices()
        .unwrap()
        .filter(|p| p.supported_extensions().contains(&device_extensions))
        .filter_map(|p| {
            p.queue_family_properties()
                .iter()
                .position(|q| q.queue_flags.intersects(QueueFlags::COMPUTE))
                .map(|i| (p, i as u32))
        })
        .min_by_key(|(p, _)| match p.properties().device_type {
            PhysicalDeviceType::DiscreteGpu => 0,
            PhysicalDeviceType::IntegratedGpu => 1,
            PhysicalDeviceType::VirtualGpu => 2,
            PhysicalDeviceType::Cpu => 3,
            PhysicalDeviceType::Other => 4,
            _ => 5,
        })
        .unwrap();

    println!(
        "Using Device: {} (Type: {:?})",
        physical_device.properties().device_name,
        physical_device.properties().device_type,
    );

    if !physical_device.supported_features().contains(&MINIMAL_FEATURES) {
        panic!("Physical device has insufficient features for this application.");
    }

    let (device, mut queues) = Device::new(
        physical_device,
        DeviceCreateInfo {
            enabled_extensions: device_extensions,
            queue_create_infos: vec![QueueCreateInfo {
                queue_family_index,
                ..Default::default()
            }],
            ..Default::default()
        }
    )
    .unwrap();

    let queue = queues.next().unwrap();

    let pipeline = {
        mod cs {
            vulkano_shaders::shader! {
                ty: "compute",
                src: r"
                    #version 450

                    // Define the dimensions of the image
                    #define WIDTH 8192
                    #define HEIGHT 8192
                    #define DEPTH 3

                    layout(local_size_x = 64, local_size_y = 1, local_size_z = 1) in;

                    layout(std430, set = 0, binding = 0) buffer Data {
                        uint data[WIDTH * HEIGHT * DEPTH];
                    };

                    struct Pixel {
                        uint r;
                        uint g;
                        uint b;
                        uint i;
                    };

                    vec3 hsv_to_rgb (vec3 c) {
                        vec4 K = vec4(1.0, 2.0 / 3.0, 1.0 / 3.0, 3.0);
                        vec3 p = abs(fract(c.xxx + K.xyz) * 6.0 - K.www);
                        return c.z * mix(K.xxx, clamp(p - K.xxx, 0.0, 1.0), c.y);
                    }

                    int from_decimal (float n) {
                        return int(mod(n, 1) * 255);
                    }

                    void main() {
                        uint idx = gl_GlobalInvocationID.x;

                        vec3 res = hsv_to_rgb(vec3(float(mod(idx, 360 * 13)) / 13, 1.0, 1.0));
                        data[DEPTH * idx] = from_decimal(res.x);
                        data[DEPTH * idx + 1] = from_decimal(res.y);
                        data[DEPTH * idx + 2] = from_decimal(res.z);
                    }
                ",
            }
        }
        let cs = cs::load(device.clone())
            .unwrap()
            .entry_point("main")
            .unwrap();

        let stage = PipelineShaderStageCreateInfo::new(cs);
        let layout = PipelineLayout::new(
            device.clone(),
            PipelineDescriptorSetLayoutCreateInfo::from_stages([&stage])
                .into_pipeline_layout_create_info(device.clone())
                .unwrap(),
        )
        .unwrap();

        ComputePipeline::new(
            device.clone(),
            None,
            ComputePipelineCreateInfo::stage_layout(stage, layout),
        ).unwrap()
    };

    let memory_allocator = Arc::new(StandardMemoryAllocator::new_default(device.clone()));

    let descriptor_set_allocator = Arc::new(StandardDescriptorSetAllocator::new(
        device.clone(),
        Default::default(),
    ));

    let command_buffer_allocator = Arc::new(StandardCommandBufferAllocator::new(
        device.clone(),
        Default::default(),
    ));

    let min_dynamic_align = device
        .physical_device()
        .properties()
        .min_uniform_buffer_offset_alignment
        .as_devicesize() as usize;
    println!("Min uniform buffer offset align: {min_dynamic_align}");

    let align = (size_of::<u8>() + min_dynamic_align - 1) & !(min_dynamic_align - 1);
    let aligned_data: Vec<u8> = Vec::with_capacity(align);

    let data_buffer = Buffer::from_iter(
        memory_allocator.clone(),
        BufferCreateInfo {
            usage: BufferUsage::STORAGE_BUFFER,
            ..Default::default()
        },
        AllocationCreateInfo {
            memory_type_filter:
                MemoryTypeFilter::PREFER_DEVICE // Tries to use GPU Memory
                | MemoryTypeFilter::HOST_RANDOM_ACCESS // Then host RAM
                | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE // Then host Storage (for the big buffers)
                ,
            ..Default::default()
        },
        aligned_data
    ).unwrap();

    // https://github.com/vulkano-rs/vulkano/blob/master/examples/dynamic-buffers/main.rs

    let layout = &pipeline.layout().set_layouts()[0];
    let set = PersistentDescriptorSet::new(
        &descriptor_set_allocator,
        layout.clone(),
        [WriteDescriptorSet::buffer(0, data_buffer.clone())],
        [],
    )
    .unwrap();

    let mut builder = AutoCommandBufferBuilder::primary(
        &command_buffer_allocator,
        queue_family_index,
        CommandBufferUsage::OneTimeSubmit,
    )
    .unwrap();

    builder
        .bind_pipeline_compute(pipeline.clone())
        .unwrap()
        .bind_descriptor_sets(
            PipelineBindPoint::Compute,
            pipeline.layout().clone(),
            0,
            set,
        )
        .unwrap();

    builder.dispatch([image_width * image_height, 1, 32]).unwrap();

    let command_buffer = builder.build().unwrap();
    let future = sync::now(device)
        .then_execute(queue, command_buffer)
        .unwrap()
        .then_signal_fence_and_flush()
        .unwrap();

    future.wait(None).unwrap();
    let data_buffer_content = data_buffer.read().unwrap();

    let data: Vec<Vec<Vec<u8>>> = data_buffer_content
        .iter()
        .filter(|v| **v <= 256)
        .map(|v| *v as u8).collect::<Vec<u8>>() // Casts to Vec<u8>
        .chunks(3) // Chunks into 3's
        .map(|pixel| Vec::from(pixel)).collect::<Vec<Vec<u8>>>() // Casts to <Vec<Vec<u8>> Where
                                                                 // inner has length 3
        .chunks(image_width as usize) // Chunks again
        .map(|row| Vec::from(row)).collect::<Vec<Vec<Vec<u8>>>>() // Collects into vec
        ;

    let mut img = RgbImage::new(image_width, image_height);
    for (i, row) in data.iter().enumerate() {
        for (j, pixel) in row.iter().enumerate() {
            if pixel.len() >= 3 && j < image_width as usize && i < image_height as usize {
                img.put_pixel(j as u32, i as u32, Rgb([pixel[0], pixel[1], pixel[2]]));
            } else {
                println!("j : {} & i : {} & Pixel : {:?}", j, i, pixel);
            }
        }
    }

    img.save("image.png").unwrap();

    // println!("{:?}", data);
    println!("{:?}", data_buffer_content.iter());

    println!("Success!");
}
