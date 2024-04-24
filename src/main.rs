extern crate vulkano;
extern crate winit;
extern crate image;
extern crate log;

use image::{
    ImageBuffer,
    Rgba,
};
use core::panic;
use std::{
    mem::size_of,
    time::Instant,
    env,
};

use vulkano::{
    buffer::{
        allocator::{SubbufferAllocator, SubbufferAllocatorCreateInfo}, BufferUsage, Subbuffer
    }, command_buffer::{
        allocator::StandardCommandBufferAllocator, AutoCommandBufferBuilder, CommandBufferUsage, CopyImageToBufferInfo
    }, descriptor_set::{
        allocator::StandardDescriptorSetAllocator,
        persistent::PersistentDescriptorSet,
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
        MemoryTypeFilter,
        StandardMemoryAllocator,
    }, pipeline::{
        compute::ComputePipelineCreateInfo, layout::PipelineDescriptorSetLayoutCreateInfo, ComputePipeline, Pipeline, PipelineBindPoint, PipelineLayout, PipelineShaderStageCreateInfo
    }, sync::{
        self,
        GpuFuture
    }, VulkanLibrary
};

use std::sync::Arc;

const MINIMAL_FEATURES: Features = Features {
    geometry_shader: true,
    ..Features::empty()
};

fn main() {

    let factor = {

        let arg: String = match env::args().nth(1) {
            Some(v) => v,
            None => "".into(),
        };

        match arg.parse::<u32>() {
            Ok(value) => value,
            Err(_) => {
                panic!("Failed to parse number from cli arg: '{arg}'!");
            }
        }
    };

    let image_width = 2u32.pow(factor);
    let image_height = 2u32.pow(factor);

    let now = Instant::now();

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
            src: "
                #version 460

                layout(local_size_x = 16, local_size_y = 16, local_size_z = 1) in;

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

                void write_data(vec3 res) {
                    imageStore(Data, ivec2(gl_GlobalInvocationID.xy), vec4(res, 1.0));
                }

                void main() {
                    // vec2 idx = gl_GlobalInvocationID.xy;
                    vec2 cords = (gl_GlobalInvocationID.xy + vec2(0.5)) / vec2(imageSize(Data));
                    vec2 c = (cords - vec2(0.5)) * 4.0;
                    vec2 z = vec2(0.0, 0.0);
                    float i;
                    float maxi = 1000.0;
                    float added = 1.0 / maxi;

                    vec3 res;

                    if (length(c) > 4.0) {
                        write_data(vec3(1.0, 1.0, 1.0));
                        return;
                    }

                    for (i = 0.0; i < 1.0; i += added) {
                        z = vec2(
                            z.x * z.x - z.y * z.y + c.x,
                            z.y * z.x + z.x * z.y + c.y
                        );

                        if (length(z) > 4.0) {
                            write_data(hsv_to_rgb(vec3(
                                mod(i * maxi * 9.0 / 360.0, 1.0),
                                1.0,
                                1.0
                            )));
                            return;
                        }
                    }

                    write_data(vec3(0.0, 0.0, 0.0));
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

    let buffer_allocator = SubbufferAllocator::new(
        memory_allocator.clone(),
        SubbufferAllocatorCreateInfo {
            buffer_usage: BufferUsage::TRANSFER_DST,
            memory_type_filter: MemoryTypeFilter::PREFER_HOST
                | MemoryTypeFilter::HOST_RANDOM_ACCESS
                | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
            ..Default::default()
        },
    );

    let buf_length = image_height as u64 * image_width as u64 * 4;

    let data_buffer: Subbuffer<[u8]> = match buffer_allocator
        .allocate_unsized(buf_length) {
            Ok(v) => v,
            Err(_) => {
                panic!("Unable to allocate '{buf_length}'!");
            },
    }
    ;

    let layout = &pipeline.layout().set_layouts()[0];
    let set = PersistentDescriptorSet::new(
        &descriptor_set_allocator,
        layout.clone(),
        [
            WriteDescriptorSet::image_view(0, view.clone()),
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
        .dispatch([image_height / 16, image_width / 16, 1])
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

    println!("Dispatch Complete");
    println!("Reading Buffer Content: {:.2?}", now.elapsed());
    let data_buffer_content = data_buffer.read().unwrap();
    println!("Buffer Read: {:.2?}", now.elapsed());

    let img = ImageBuffer::<Rgba<u8>, _>::from_raw(image_width, image_height, &data_buffer_content[..]).unwrap();
    println!("Saving file...");
    img.save("image.png").unwrap();
    println!("Image Saved: {:.2?}", now.elapsed());

    println!("Success!: {:.2?} ({image_width} x {image_height})", now.elapsed());
}
