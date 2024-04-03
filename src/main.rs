extern crate vulkano;
extern crate winit;

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
        panic!("The physical device is not good enough for this application");
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

                    Pixel hsv_to_rgb(vec3 hsv) {
                        // https://stackoverflow.com/a/26856771/15474643
                        // I used the python version of hsv_to_rgb but converted to glsl

                        float h = hsv.x;
                        float s = hsv.y;
                        float v = hsv.z;

                        uint i = int(h * 6.0);
                        float f = h * 6.0 - float(i);

                        float w = v * (1.0 - s);
                        float q = v * (1.0 - s * f);
                        float t = v * (1.0 - s * (1.0 - f));

                        i = int(mod(i, 6));

                        float r, g, b;

                        if (i == 0) {
                            r = v;
                            g = t;
                            b = w;
                        } else if (i == 1) {
                            r = q;
                            g = v;
                            b = w;
                        } else if (i == 2) {
                            r = w;
                            g = v;
                            b = t;
                        } else if (i == 3) {
                            r = w;
                            g = q;
                            b = v;
                        } else if (i == 4) {
                            r = t;
                            g = w;
                            b = v;
                        } else if (i == 5) {
                            r = v;
                            g = w;
                            b = q;
                        } else { // (i < 6.0)
                            r = 21.0;
                            g = 22.0;
                            b = 23.0;
                        }

                        Pixel result;
                        result.r = int(r * 255.0);
                        result.g = int(g * 255.0);
                        result.b = int(b * 255.0);

                        return result;
                    }

                    void main() {
                        uint idx = gl_GlobalInvocationID.x;
                        if (idx < WIDTH * HEIGHT * DEPTH) {
                            Pixel res = hsv_to_rgb(vec3(float(idx) / 13, 1.0, 1.0));
                            data[DEPTH * idx] = res.r;
                            data[DEPTH * idx + 1] = res.g;
                            data[DEPTH * idx + 2] = res.b;
                            data[DEPTH * idx + 3] = res.i;
                        }
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

    let memory_allocator= Arc::new(StandardMemoryAllocator::new_default(device.clone()));
    let descriptor_set_allocator = Arc::new(StandardDescriptorSetAllocator::new(
        device.clone(),
        Default::default(),
    ));
    let command_buffer_allocator = Arc::new(StandardCommandBufferAllocator::new(
        device.clone(),
        Default::default(),
    ));
    let data_buffer = Buffer::from_iter(
        memory_allocator,
        BufferCreateInfo {
            usage: BufferUsage::STORAGE_BUFFER,
            ..Default::default()
        },
        AllocationCreateInfo {
            memory_type_filter: MemoryTypeFilter::PREFER_DEVICE | MemoryTypeFilter::HOST_RANDOM_ACCESS,
            ..Default::default()
        },
        0..65536u32,
    )
    .unwrap();

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

    builder.dispatch([1024, 1, 1]).unwrap();

    let command_buffer = builder.build().unwrap();
    let future = sync::now(device)
        .then_execute(queue, command_buffer)
        .unwrap()
        .then_signal_fence_and_flush()
        .unwrap();

    future.wait(None).unwrap();
    let data_buffer_content = data_buffer.read().unwrap();
    let mut data: u32;
    for n in 0..50u32 {
        data = data_buffer_content[n as usize];
        println!("{:?}", data);
        // assert_eq!(data_buffer_content[n as usize], n * 12);
    }

    println!("Success!");
}

