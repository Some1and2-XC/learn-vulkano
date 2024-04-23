extern crate vulkano;
extern crate winit;
extern crate image;

use image::{RgbImage, Rgb};
use std::{
    io,
    mem::size_of,
    iter::repeat,
};

use vulkano::{
    buffer::{
        Buffer,
        BufferCreateInfo,
        BufferUsage,
    }, command_buffer::{
        allocator::StandardCommandBufferAllocator, sys::CommandBufferBeginInfo, AutoCommandBufferBuilder, CommandBufferLevel, CommandBufferUsage, CopyImageToBufferInfo
    }, descriptor_set::{
        allocator::StandardDescriptorSetAllocator,
        layout::DescriptorType,
        persistent::PersistentDescriptorSet,
        DescriptorBufferInfo,
        DescriptorSet,
        WriteDescriptorSet,
    }, device::{
        physical::PhysicalDeviceType,
        Device,
        DeviceCreateInfo,
        DeviceExtensions,
        Features,
        QueueCreateInfo,
        QueueFlags,
    }, format::Format, image::{
        view::ImageView, Image, ImageCreateInfo, ImageType, ImageUsage
    }, instance::{
        Instance,
        InstanceCreateFlags,
        InstanceCreateInfo,
    }, memory::allocator::{
        AllocationCreateInfo,
        DeviceLayout,
        MemoryTypeFilter,
        StandardMemoryAllocator,
    }, pipeline::{
        self, compute::ComputePipelineCreateInfo, layout::PipelineDescriptorSetLayoutCreateInfo, ComputePipeline, Pipeline, PipelineBindPoint, PipelineLayout, PipelineShaderStageCreateInfo
    }, sync::{
        self,
        GpuFuture
    }, DeviceSize, VulkanLibrary
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
    let image_bytes = image_width * image_height * 4;

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
                .map(|i| (p, i.try_into().unwrap()))
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

    mod cs {
        vulkano_shaders::shader! {
            ty: "compute",
            src: r"
                #version 460

                layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;

                // layout(std430, set = 0, binding = 0) buffer InData {
                //     uint index;
                // } ub;

                layout(set = 0, binding = 0, rgba8) uniform writeonly image2D Data;

                vec3 hsv_to_rgb (vec3 c) {
                    vec4 K = vec4(1.0, 2.0 / 3.0, 1.0 / 3.0, 3.0);
                    vec3 p = abs(fract(c.xxx + K.xyz) * 6.0 - K.www);
                    return c.z * mix(K.xxx, clamp(p - K.xxx, 0.0, 1.0), c.y);
                }

                int from_decimal (float n) {
                    return int(mod(n, 1) * 255);
                }

                void main() {
                    vec2 idx = gl_GlobalInvocationID.xy;

                    vec3 res = hsv_to_rgb(
                        vec3(
                            mod(idx.x / idx.y, 1.0),
                            1.0,
                            1.0
                        )
                    );

                    imageStore(Data, ivec2(idx), vec4(res, 1.0));
                    // data[idx.x] = res;
                }
            ",
        }
    }

    let pipeline = {
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
            ).unwrap();

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

    let image = Image::new(
        memory_allocator.clone(),
        ImageCreateInfo {
            image_type: ImageType::Dim2d,
            format: Format::R8G8B8A8_UNORM,
            extent: [image_width, image_height, 1],
            usage: ImageUsage::STORAGE | ImageUsage::TRANSFER_SRC,
            ..Default::default()
        },
        AllocationCreateInfo {
            memory_type_filter: MemoryTypeFilter::PREFER_DEVICE,
            ..Default::default()
        },
    ).unwrap();

    let view = ImageView::new_default(image.clone()).unwrap();

    let data_buffer = Buffer::from_iter(
        memory_allocator.clone(),
        BufferCreateInfo {
            usage: BufferUsage::TRANSFER_DST,
            ..Default::default()
        },
        AllocationCreateInfo {
            memory_type_filter: MemoryTypeFilter::PREFER_HOST
                | MemoryTypeFilter::HOST_RANDOM_ACCESS
                | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
            ..Default::default()
        },
        (0..(image_width * image_height * 4)).map(|_| 0u8),
    ).unwrap();

    // https://github.com/vulkano-rs/vulkano/blob/master/examples/dynamic-buffers/main.rs

    let layout = &pipeline.layout().set_layouts()[0];
    let set = PersistentDescriptorSet::new(
        &descriptor_set_allocator,
        layout.clone(),
        [
            WriteDescriptorSet::image_view(0, view.clone()),
            // WriteDescriptorSet::buffer_with_range(
            //     0,
            //     DescriptorBufferInfo {
            //         buffer: input_data,
            //         range: 0..size_of::<cs::InData>() as DeviceSize,
            //     },
            // ),
        ],
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
        .unwrap()
        ;

    builder
        .dispatch([image_width / 8, image_height / 8, 1])
        .unwrap()
        .copy_image_to_buffer(CopyImageToBufferInfo::image_buffer(
            image.clone(),
            data_buffer.clone(),
        ))
        .unwrap()
        ;

    let command_buffer = builder.build().unwrap();
    let future = sync::now(device)
        .then_execute(queue, command_buffer)
        .unwrap()
        .then_signal_fence_and_flush()
        .unwrap();
    future.wait(None).unwrap();

    let data_buffer_content = data_buffer
        .read()
        .unwrap()
        ;

    let data: Vec<Vec<Vec<u8>>> = data_buffer_content
        .iter()
        .map(|v| *v as u8).collect::<Vec<u8>>()
        // .filter(|v| **v <= 256)
        // .map(|v| *v as u8).collect::<Vec<u8>>() // Casts to Vec<u8>
        .chunks(4) // Chunks into 3's
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
    // println!("{:?}",
    //      data_buffer_content
    //          .chunks(4)
    //          .map(|v| Vec::from(v))
    //          .collect::<Vec<Vec<f32>>>()
    //      );
    println!("Success!");
}
