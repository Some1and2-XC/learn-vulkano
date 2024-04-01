extern crate vulkano;
extern crate winit;

use std::sync::Arc;

use winit::event_loop::{EventLoop, EventLoopBuilder};
use winit::window::WindowBuilder;

use vulkano::{
    instance::{
        Instance,
        InstanceExtensions,
        InstanceCreateInfo,
    },
    device::{
        Device,
        DeviceCreateInfo,
        physical::PhysicalDevice,
        DeviceExtensions,
        QueueFlags,
    },
    swapchain::Surface,
    Version, VulkanLibrary,
};

// use std::{time, thread};

struct HelloTriangleApplication {
    event_loop: EventLoop<()>,
    instance: Arc<Instance>,
}

impl HelloTriangleApplication {
    pub fn new() -> Self {

        let library = VulkanLibrary::new()
            .unwrap_or_else(|err| panic!("Couldn't load Vulkan library: {:?}", err));

        let layers: Vec<_> = library.layer_properties().unwrap().collect();

        let extensions = InstanceExtensions {
            khr_surface: true,
            khr_android_surface: false,
            ..InstanceExtensions::empty()
        };

        let instance = Instance::new(
            library,
            InstanceCreateInfo {
                enabled_extensions: extensions,
                enabled_layers: layers.iter().map(|l| l.name().to_owned()).collect(),
                ..Default::default()
            }
        ).unwrap_or_else(|err| panic!("Couldn't create instance: {:?}", err));

        let physical = instance
            .enumerate_physical_devices()
            .unwrap()
            .next()
            .unwrap_or_else(|| panic!("Couldn't get physical device"));

        let event_loop = EventLoopBuilder::new()
            .build()
            .unwrap();

        let surface = WindowBuilder::new()
            .with_title("Learning Vulkan")
            .build(&event_loop)
            .unwrap();
            // .build(&event_loop, instance.clone());

        let queue_family = physical
            .queue_family_properties();

        let device_ext = DeviceExtensions {
            khr_swapchain: true,
            ..DeviceExtensions::none()
        };

        let create_device_info = CreateDeviceInfo::new

        let (device, mut queues) = Device::new(
            physical,
            DeviceCreateInfo::new(),
        )
        .unwrap();

        Self {
            instance,
            event_loop,
        }
    }

    fn main_loop(&mut self) {
        let mut done = false;
        // let interval = time::Duration::from_secs(1);
        // thread::sleep(interval);
        // while !done {
        // }
    }
}

fn main() {
    println!("Starting Main Thing");
    let mut app = HelloTriangleApplication::new();
    app.main_loop();
    println!("Finished Main Thing");
}
